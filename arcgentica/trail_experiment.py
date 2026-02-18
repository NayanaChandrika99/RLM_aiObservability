# ABOUTME: Orchestrates TRAIL benchmark experiments with built-in scoring and experiment logging.
# ABOUTME: Runs analyze_trace over trace datasets, scores against gold annotations, and writes metrics.

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

try:
    from .trail_agent import analyze_trace
    from .trail_common import TRAIL_LEAF_CATEGORIES
    from .trail_semantic_checks import enforce_semantic_faithfulness
except ImportError:
    from trail_agent import analyze_trace
    from trail_common import TRAIL_LEAF_CATEGORIES
    from trail_semantic_checks import enforce_semantic_faithfulness


@dataclass
class ExperimentConfig:
    experiment_id: str
    trail_data_dir: Path
    gold_dir: Path
    output_dir: Path
    model: str = "openai/gpt-5-mini"
    root_model: str | None = None
    chunk_model: str | None = None
    approach: str = "on"
    subset: str = "dev18"
    prompt_version: str = "v2"
    split: str = "GAIA"
    max_workers: int = 5
    max_chunks: int = 6
    max_num_agents: int = 6
    agent_call_timeout_seconds: float = 60.0
    agent_timeout_retries: int = 1
    joint_recall_boost: bool = False
    semantic_checks: str = "strict"
    resume: bool = True
    notes: str = ""


def load_subset_trace_ids(subset: str) -> list[str] | None:
    """Return trace IDs for a named subset, or None for 'full'."""
    if subset == "full":
        return None
    if subset == "dev18":
        manifest_path = Path(__file__).resolve().parent / "dev_subset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["trace_ids"]
    raise ValueError(f"Unknown subset: {subset!r}. Expected 'full' or 'dev18'.")


def _preflight_repl_environment(config: ExperimentConfig) -> None:
    if config.approach != "on":
        raise ValueError("REPL-only mode is enabled. Use --approach on.")

    try:
        importlib.import_module("agentica")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "REPL mode requires 'agentica'. Run the experiment from the arcgentica environment."
        ) from exc


def _progress_entry_path(progress_dir: Path, trace_stem: str) -> Path:
    return progress_dir / f"{trace_stem}.json"


def _normalize_category(category: str, all_categories: list[str]) -> str:
    """Normalize a category name by finding the closest match in all_categories."""
    if not category:
        return ""

    cat_lower = category.lower().strip()
    cat_no_spaces = cat_lower.replace(" ", "")

    for std_cat in all_categories:
        if cat_lower == std_cat.lower() or cat_no_spaces == std_cat.lower().replace(" ", ""):
            return std_cat

    # Substring matching
    for std_cat in all_categories:
        if cat_no_spaces in std_cat.lower().replace(" ", ""):
            return std_cat

    return category


def _calculate_per_trace_metrics(
    ground_truth: dict[str, Any],
    generated: dict[str, Any],
    all_categories: list[str],
) -> dict[str, Any]:
    """Compute per-trace metrics matching the official scorer logic."""
    gt_errors = ground_truth.get("errors", [])
    gt_categories_raw = [e.get("category", "") for e in gt_errors]
    gt_locations = [e.get("location", "") for e in gt_errors]

    gen_errors = generated.get("errors", [])
    gen_categories_raw = [e.get("category", "") for e in gen_errors]
    gen_locations = [e.get("location", "") for e in gen_errors]

    gt_categories = [_normalize_category(c, all_categories) for c in gt_categories_raw if c]
    gen_categories = [_normalize_category(c, all_categories) for c in gen_categories_raw if c]

    # Location-category joint accuracy (set intersection)
    gt_loc_cat_pairs = [
        (gt_locations[i], gt_categories[i])
        for i in range(len(gt_locations))
        if i < len(gt_categories)
    ]
    gen_loc_cat_pairs = [
        (gen_locations[i], gen_categories[i])
        for i in range(len(gen_locations))
        if i < len(gen_categories)
    ]
    common_pairs = set(gt_loc_cat_pairs).intersection(set(gen_loc_cat_pairs))
    joint_accuracy = len(common_pairs) / len(set(gt_loc_cat_pairs)) if gt_loc_cat_pairs else 0

    # Location accuracy (set intersection)
    common_locations = set(gt_locations).intersection(set(gen_locations))
    location_accuracy = len(common_locations) / len(set(gt_locations)) if gt_locations else 0

    # Binary vectors for multi-label F1
    y_true = np.zeros(len(all_categories))
    y_pred = np.zeros(len(all_categories))
    for cat in gt_categories:
        if cat in all_categories:
            y_true[all_categories.index(cat)] = 1
    for cat in gen_categories:
        if cat in all_categories:
            y_pred[all_categories.index(cat)] = 1

    # Extract scores
    gt_scores = ground_truth.get("scores", [{}])[0] if ground_truth.get("scores") else {}
    gen_scores = generated.get("scores", [{}])[0] if generated.get("scores") else {}

    return {
        "location_accuracy": location_accuracy,
        "joint_accuracy": joint_accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "gt_scores": gt_scores,
        "gen_scores": gen_scores,
    }


def _safe_float(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_CLUSTER_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_.]{2,}")
_CLUSTER_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "when",
    "where",
    "what",
    "which",
    "were",
    "was",
    "had",
    "has",
    "have",
    "will",
    "would",
    "should",
    "could",
    "can",
    "did",
    "does",
    "done",
    "not",
    "then",
    "than",
    "their",
    "them",
    "they",
    "only",
    "same",
    "tool",
    "error",
    "errors",
    "issue",
    "issues",
    "trace",
    "span",
    "spans",
    "agent",
    "analysis",
    "failed",
    "failure",
}


def _json_loads_relaxed(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", raw_text)
        return json.loads(repaired)


def _cluster_tokens(value: str) -> set[str]:
    normalized = " ".join(value.lower().split())
    tokens: set[str] = set()
    for token in _CLUSTER_TOKEN_PATTERN.findall(normalized):
        if token in _CLUSTER_STOPWORDS or token.isdigit():
            continue
        tokens.add(token)
        if "." in token:
            suffix = token.split(".")[-1]
            if suffix and suffix not in _CLUSTER_STOPWORDS:
                tokens.add(suffix)
    return tokens


def _error_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["category"] != right["category"]:
        return 0.0

    left_tokens: set[str] = left["tokens"]
    right_tokens: set[str] = right["tokens"]
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union == 0:
        return 0.0
    jaccard = overlap / union
    if overlap >= 2:
        return max(jaccard, 0.35)
    return jaccard


def _build_cross_trace_cluster_report(generated_dir: Path) -> dict[str, Any]:
    error_rows: list[dict[str, Any]] = []
    processed_files = 0

    for output_file in sorted(generated_dir.glob("*.json")):
        try:
            payload = _json_loads_relaxed(output_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        processed_files += 1
        trace_id = str(payload.get("trace_id", "")) or output_file.stem
        errors = payload.get("errors", [])
        if not isinstance(errors, list):
            continue

        for idx, error in enumerate(errors):
            if not isinstance(error, dict):
                continue
            category = error.get("category")
            location = error.get("location")
            if not isinstance(category, str) or not category.strip():
                continue
            if not isinstance(location, str) or not location.strip():
                continue
            evidence = error.get("evidence", "")
            description = error.get("description", "")
            normalized_description = " ".join(str(description).split()).lower() if isinstance(description, str) else ""
            if normalized_description.startswith("co-located "):
                continue
            text = f"{evidence if isinstance(evidence, str) else ''} {description if isinstance(description, str) else ''}".strip()
            tokens = _cluster_tokens(text)
            error_rows.append(
                {
                    "id": f"{trace_id}:{idx}",
                    "trace_id": trace_id,
                    "category": category.strip(),
                    "location": location.strip(),
                    "evidence": str(evidence)[:300] if isinstance(evidence, str) else "",
                    "description": str(description)[:300] if isinstance(description, str) else "",
                    "tokens": tokens,
                }
            )

    n = len(error_rows)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            similarity = _error_similarity(error_rows[i], error_rows[j])
            if similarity >= 0.30:
                adjacency[i].add(j)
                adjacency[j].add(i)

    clusters_raw: list[list[int]] = []
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        component.sort()
        clusters_raw.append(component)

    def _cluster_sort_key(component: list[int]) -> tuple[int, str, str]:
        first_row = error_rows[component[0]]
        return (-len(component), first_row["category"], first_row["trace_id"])

    clusters_raw.sort(key=_cluster_sort_key)

    clusters: list[dict[str, Any]] = []
    for idx, component in enumerate(clusters_raw, start=1):
        rows = [error_rows[i] for i in component]
        category_counter = Counter(row["category"] for row in rows)
        dominant_category = sorted(
            category_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        token_counter: Counter[str] = Counter()
        for row in rows:
            token_counter.update(row["tokens"])
        token_signature = [token for token, _count in token_counter.most_common(6)]

        trace_ids = sorted({row["trace_id"] for row in rows})
        locations = sorted({f"{row['trace_id']}:{row['location']}" for row in rows})
        sample_errors = [
            {
                "trace_id": row["trace_id"],
                "category": row["category"],
                "location": row["location"],
                "evidence": row["evidence"],
                "description": row["description"],
            }
            for row in rows[:3]
        ]

        clusters.append(
            {
                "cluster_id": f"cluster_{idx:03d}",
                "size": len(rows),
                "trace_count": len(trace_ids),
                "dominant_category": dominant_category,
                "category_distribution": dict(sorted(category_counter.items(), key=lambda item: item[0])),
                "trace_ids": trace_ids,
                "locations": locations,
                "token_signature": token_signature,
                "sample_errors": sample_errors,
            }
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "files_processed": processed_files,
        "total_errors": len(error_rows),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }


def _score_outputs(
    gold_dir: Path,
    generated_dir: Path,
    trace_files: list[Path],
) -> dict[str, Any]:
    """Built-in scorer matching official calculate_scores.py logic."""
    all_categories = list(TRAIL_LEAF_CATEGORIES)

    location_accuracy_sum = 0.0
    joint_accuracy_sum = 0.0
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    # Score correlation accumulators
    score_keys = [
        ("reliability_score", "reliability"),
        ("security_score", "security"),
        ("instruction_adherence_score", "instruction_adherence"),
        ("plan_opt_score", "plan_optimization"),
        ("overall", "overall"),
    ]
    gt_score_lists: dict[str, list[float]] = {name: [] for _, name in score_keys}
    gen_score_lists: dict[str, list[float]] = {name: [] for _, name in score_keys}

    files_processed = 0

    for trace_file in trace_files:
        trace_id = trace_file.stem
        gold_path = gold_dir / f"{trace_id}.json"
        gen_path = generated_dir / f"{trace_id}.json"

        if not gold_path.exists() or not gen_path.exists():
            continue

        try:
            ground_truth = _json_loads_relaxed(gold_path.read_text(encoding="utf-8"))
            generated = _json_loads_relaxed(gen_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metrics = _calculate_per_trace_metrics(ground_truth, generated, all_categories)

        all_y_true.append(metrics["y_true"])
        all_y_pred.append(metrics["y_pred"])
        location_accuracy_sum += metrics["location_accuracy"]
        joint_accuracy_sum += metrics["joint_accuracy"]

        gt_scores = metrics["gt_scores"]
        gen_scores = metrics["gen_scores"]
        if gt_scores and gen_scores:
            for field_key, name in score_keys:
                if field_key in gt_scores and field_key in gen_scores:
                    gt_val = _safe_float(gt_scores.get(field_key))
                    gen_val = _safe_float(gen_scores.get(field_key))
                    gt_score_lists[name].append(gt_val)
                    gen_score_lists[name].append(gen_val)

        files_processed += 1

    if files_processed == 0:
        return {
            "weighted_f1": 0.0,
            "location_accuracy": 0.0,
            "joint_accuracy": 0.0,
            "category_metrics": {cat: {"precision": 0, "recall": 0, "f1": 0, "support": 0} for cat in all_categories},
            "score_correlations": {},
            "files_processed": 0,
        }

    location_accuracy_avg = location_accuracy_sum / files_processed
    joint_accuracy_avg = joint_accuracy_sum / files_processed

    # Score correlations (Pearson r)
    score_correlations: dict[str, Any] = {}
    for _, name in score_keys:
        gt_list = gt_score_lists[name]
        gen_list = gen_score_lists[name]
        if len(gt_list) < 2 or len(gen_list) < 2:
            continue
        if len(set(gt_list)) < 2 or len(set(gen_list)) < 2:
            continue
        try:
            corr, p_value = pearsonr(gt_list, gen_list)
        except Exception:
            continue
        score_correlations[name] = {
            "correlation": float(corr) if not np.isnan(corr) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "n": len(gt_list),
        }

    # Aggregate for weighted F1 and per-category metrics
    all_y_true_array = np.vstack(all_y_true)
    all_y_pred_array = np.vstack(all_y_pred)

    category_metrics: dict[str, Any] = {}
    for i, category in enumerate(all_categories):
        true_positives = int(np.sum((all_y_true_array[:, i] == 1) & (all_y_pred_array[:, i] == 1)))
        false_positives = int(np.sum((all_y_true_array[:, i] == 0) & (all_y_pred_array[:, i] == 1)))
        false_negatives = int(np.sum((all_y_true_array[:, i] == 1) & (all_y_pred_array[:, i] == 0)))
        support = int(np.sum(all_y_true_array[:, i]))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    weighted_f1 = float(f1_score(all_y_true_array, all_y_pred_array, average="weighted", zero_division=0))

    return {
        "weighted_f1": weighted_f1,
        "location_accuracy": location_accuracy_avg,
        "joint_accuracy": joint_accuracy_avg,
        "category_metrics": category_metrics,
        "score_correlations": score_correlations,
        "files_processed": files_processed,
    }


def _process_trace_file(
    trace_file: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    try:
        trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "trace_file": trace_file.name,
            "trace_stem": trace_file.stem,
            "status": "failed_to_read_trace",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:300],
        }

    try:
        prediction = analyze_trace(
            trace_payload,
            model=config.model,
            root_model=config.root_model,
            chunk_model=config.chunk_model,
            agentic_mode=config.approach,
            prompt_version=config.prompt_version,
            max_chunks=config.max_chunks,
            max_num_agents=config.max_num_agents,
            agent_call_timeout_seconds=config.agent_call_timeout_seconds,
            agent_timeout_retries=config.agent_timeout_retries,
            joint_recall_boost=config.joint_recall_boost,
        )
    except Exception as exc:
        return {
            "trace_file": trace_file.name,
            "trace_stem": trace_file.stem,
            "status": "failed_to_analyze_trace",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:300],
        }

    repaired_prediction, semantic_report = enforce_semantic_faithfulness(
        trace_payload, prediction, mode=config.semantic_checks
    )
    return {
        "trace_file": trace_file.name,
        "trace_stem": trace_file.stem,
        "status": "ok",
        "prediction": repaired_prediction,
        "semantic_report": semantic_report,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a full TRAIL experiment: analyze traces, score, write outputs."""
    _preflight_repl_environment(config)

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()

    # Create output directory
    exp_dir = config.output_dir / config.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict: dict[str, Any] = {}
    for k, v in asdict(config).items():
        config_dict[k] = str(v) if isinstance(v, Path) else v
    (exp_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # Load trace files
    split_dir = config.trail_data_dir / config.split
    if not split_dir.exists():
        raise FileNotFoundError(f"TRAIL split directory not found: {split_dir}")

    all_trace_files = sorted(split_dir.glob("*.json"))

    # Filter by subset
    subset_ids = load_subset_trace_ids(config.subset)
    if subset_ids is not None:
        subset_set = set(subset_ids)
        trace_files = [f for f in all_trace_files if f.stem in subset_set]
    else:
        trace_files = all_trace_files

    # Process traces
    results_dir = exp_dir / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = exp_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    traces_processed = 0
    traces_failed = 0
    analysis_fallbacks = 0
    delegation_failures = 0
    semantic_total_errors = 0
    semantic_kept_errors = 0
    semantic_dropped_errors = 0
    semantic_repair_actions = {
        "location_repaired": 0,
        "evidence_repaired": 0,
        "impact_repaired": 0,
        "description_repaired": 0,
    }
    semantic_drop_reasons = {
        "invalid_error_shape": 0,
        "missing_category": 0,
        "unrepairable_location": 0,
        "unrepairable_evidence": 0,
    }
    semantic_file_reports: list[dict[str, Any]] = []

    def _apply_progress_entry(progress_entry: dict[str, Any]) -> None:
        nonlocal traces_processed
        nonlocal traces_failed
        nonlocal analysis_fallbacks
        nonlocal delegation_failures
        nonlocal semantic_total_errors
        nonlocal semantic_kept_errors
        nonlocal semantic_dropped_errors

        if progress_entry.get("status") != "ok":
            traces_failed += 1
            semantic_file_reports.append(
                {
                    "trace_file": progress_entry.get("trace_file", ""),
                    "status": progress_entry.get("status", ""),
                    "error_type": progress_entry.get("error_type", ""),
                    "error_message": progress_entry.get("error_message", ""),
                }
            )
            return

        semantic_report = progress_entry.get("semantic_report", {})
        if not isinstance(semantic_report, dict):
            semantic_report = {}

        semantic_total_errors += int(semantic_report.get("total_errors", 0))
        semantic_kept_errors += int(semantic_report.get("kept_errors", 0))
        semantic_dropped_errors += int(semantic_report.get("dropped_errors", 0))
        for key in semantic_repair_actions:
            semantic_repair_actions[key] += int(semantic_report.get("repair_actions", {}).get(key, 0))
        for key in semantic_drop_reasons:
            semantic_drop_reasons[key] += int(semantic_report.get("drop_reasons", {}).get(key, 0))

        analysis_diagnostics = progress_entry.get("analysis_diagnostics")
        if isinstance(analysis_diagnostics, dict) and analysis_diagnostics.get("analysis_mode") == "heuristic_fallback":
            analysis_fallbacks += 1
        if isinstance(analysis_diagnostics, dict):
            delegation_failures += int(analysis_diagnostics.get("delegation_failures", 0))

        trace_semantic_report = dict(semantic_report)
        trace_semantic_report["trace_file"] = progress_entry.get("trace_file", "")
        trace_semantic_report["status"] = "ok"
        if isinstance(analysis_diagnostics, dict):
            trace_semantic_report["analysis_diagnostics"] = analysis_diagnostics
        semantic_file_reports.append(trace_semantic_report)
        traces_processed += 1

    def _store_trace_result(trace_result: dict[str, Any]) -> None:
        trace_stem = trace_result["trace_stem"]
        trace_file = trace_result["trace_file"]
        progress_path = _progress_entry_path(progress_dir, trace_stem)

        if trace_result["status"] == "ok":
            prediction = trace_result["prediction"]
            output_path = results_dir / f"{trace_stem}.json"
            output_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

            analysis_diagnostics = prediction.get("analysis_diagnostics")
            progress_entry: dict[str, Any] = {
                "trace_file": trace_file,
                "trace_stem": trace_stem,
                "status": "ok",
                "semantic_report": trace_result["semantic_report"],
            }
            if isinstance(analysis_diagnostics, dict):
                progress_entry["analysis_diagnostics"] = analysis_diagnostics
        else:
            progress_entry = {
                "trace_file": trace_file,
                "trace_stem": trace_stem,
                "status": trace_result["status"],
                "error_type": trace_result.get("error_type", ""),
                "error_message": trace_result.get("error_message", ""),
            }

        progress_path.write_text(json.dumps(progress_entry, indent=2), encoding="utf-8")
        _apply_progress_entry(progress_entry)

    completed_trace_stems: set[str] = set()
    if config.resume:
        trace_by_stem = {trace_file.stem: trace_file for trace_file in trace_files}
        for trace_stem, trace_file in trace_by_stem.items():
            progress_path = _progress_entry_path(progress_dir, trace_stem)
            output_path = results_dir / f"{trace_stem}.json"

            if progress_path.exists():
                try:
                    progress_entry = json.loads(progress_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if progress_entry.get("status") == "ok" and not output_path.exists():
                    continue
                _apply_progress_entry(progress_entry)
                completed_trace_stems.add(trace_stem)
                continue

            if not output_path.exists():
                continue

            try:
                trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
                prediction = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            _prediction, semantic_report = enforce_semantic_faithfulness(
                trace_payload,
                prediction if isinstance(prediction, dict) else {},
                mode=config.semantic_checks,
            )
            del _prediction

            progress_entry = {
                "trace_file": trace_file.name,
                "trace_stem": trace_stem,
                "status": "ok",
                "semantic_report": semantic_report,
            }
            analysis_diagnostics = prediction.get("analysis_diagnostics") if isinstance(prediction, dict) else None
            if isinstance(analysis_diagnostics, dict):
                progress_entry["analysis_diagnostics"] = analysis_diagnostics

            progress_path.write_text(json.dumps(progress_entry, indent=2), encoding="utf-8")
            _apply_progress_entry(progress_entry)
            completed_trace_stems.add(trace_stem)

    pending_trace_files = [trace_file for trace_file in trace_files if trace_file.stem not in completed_trace_stems]
    worker_count = max(1, int(config.max_workers))
    if worker_count <= 1:
        for trace_file in pending_trace_files:
            _store_trace_result(_process_trace_file(trace_file, config))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_process_trace_file, trace_file, config): trace_file
                for trace_file in pending_trace_files
            }
            for future in as_completed(future_map):
                trace_file = future_map[future]
                try:
                    trace_result = future.result()
                except Exception as exc:
                    trace_result = {
                        "trace_file": trace_file.name,
                        "trace_stem": trace_file.stem,
                        "status": "failed_to_analyze_trace",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    }
                _store_trace_result(trace_result)

    semantic_report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": config.semantic_checks,
        "split": config.split,
        "model": config.model,
        "totals": {
            "traces_processed": traces_processed,
            "traces_failed": traces_failed,
            "total_errors": semantic_total_errors,
            "kept_errors": semantic_kept_errors,
            "dropped_errors": semantic_dropped_errors,
            "analysis_fallbacks": analysis_fallbacks,
            "delegation_failures": delegation_failures,
            "grounded_evidence_rate": 1.0 if semantic_total_errors == 0 else semantic_kept_errors / semantic_total_errors,
            "location_repaired": semantic_repair_actions["location_repaired"],
            "evidence_repaired": semantic_repair_actions["evidence_repaired"],
        },
        "repair_actions": semantic_repair_actions,
        "drop_reasons": semantic_drop_reasons,
        "files": semantic_file_reports,
    }
    semantic_report_path = exp_dir / "semantic_report.json"
    semantic_report_path.write_text(json.dumps(semantic_report, indent=2), encoding="utf-8")

    # Score outputs against gold
    metrics = _score_outputs(config.gold_dir, results_dir, trace_files)
    cross_trace_clusters = _build_cross_trace_cluster_report(results_dir)
    cross_trace_clusters_path = exp_dir / "cross_trace_clusters.json"
    cross_trace_clusters_path.write_text(json.dumps(cross_trace_clusters, indent=2), encoding="utf-8")

    # Write metrics
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - start_time
    completed_at = datetime.now(timezone.utc).isoformat()

    # Build experiment record
    record: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "model": config.model,
        "root_model": config.root_model or config.model,
        "chunk_model": config.chunk_model or config.model,
        "approach": config.approach,
        "subset": config.subset,
        "split": config.split,
        "prompt_version": config.prompt_version,
        "semantic_checks": config.semantic_checks,
        "traces_processed": traces_processed,
        "traces_failed": traces_failed,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "semantic": {
            "report_path": str(semantic_report_path),
            "mode": config.semantic_checks,
            "totals": semantic_report["totals"],
        },
        "cross_trace_clusters": {
            "report_path": str(cross_trace_clusters_path),
            "cluster_count": int(cross_trace_clusters.get("cluster_count", 0)),
            "total_errors": int(cross_trace_clusters.get("total_errors", 0)),
        },
        "notes": config.notes,
    }

    # Append to experiment log
    log_path = config.output_dir / "experiment_log.json"
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(log_data, list):
                log_data = [log_data]
        except Exception:
            log_data = []
    else:
        log_data = []
    log_data.append(record)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_id}")
    print(
        f"Model: {config.model} "
        f"(root={config.root_model or config.model}, chunk={config.chunk_model or config.model}) "
        f"| Approach: {config.approach} | Subset: {config.subset}"
    )
    print(f"Traces processed: {traces_processed} | Failed: {traces_failed}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'-'*60}")
    print(f"Weighted F1:      {metrics.get('weighted_f1', 0):.4f}")
    print(f"Location Acc:     {metrics.get('location_accuracy', 0):.4f}")
    print(f"Joint Acc:        {metrics.get('joint_accuracy', 0):.4f}")
    if metrics.get("score_correlations"):
        print(f"{'-'*60}")
        print(f"{'Score Type':<25} {'Pearson r':<12} {'p-value':<12}")
        for score_type, corr_data in metrics["score_correlations"].items():
            print(f"{score_type:<25} {corr_data['correlation']:>10.4f}  {corr_data['p_value']:>10.4f}")
    print(f"{'='*60}\n")

    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TRAIL benchmark experiment",
    )
    parser.add_argument("--experiment-id", required=True, help="Unique experiment identifier")
    parser.add_argument("--trail-data-dir", type=Path, required=True, help="Dir containing split subfolders (e.g. GAIA/)")
    parser.add_argument("--gold-dir", type=Path, required=True, help="Dir containing gold annotation JSONs")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"), help="Output directory")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="LiteLLM model string")
    parser.add_argument("--root-model", default=None, help="Model for root planning call (defaults to --model)")
    parser.add_argument("--chunk-model", default=None, help="Model for delegated chunk calls (defaults to --model)")
    parser.add_argument("--approach", default="on", choices=["on"], help="Agentic mode")
    parser.add_argument("--subset", default="dev18", help="Subset name (dev18 or full)")
    parser.add_argument("--prompt-version", default="v2", help="Prompt version tag")
    parser.add_argument("--split", default="GAIA", help="TRAIL data split name")
    parser.add_argument("--max-workers", type=int, default=5, help="Max parallel workers")
    parser.add_argument("--max-chunks", type=int, default=6, help="Max delegated chunks analyzed per trace")
    parser.add_argument("--max-num-agents", type=int, default=6, help="Agent budget per trace (must exceed max_chunks for full chunk coverage)")
    parser.add_argument(
        "--agent-call-timeout-seconds",
        type=float,
        default=60.0,
        help="Per delegated root/chunk agent call timeout in seconds.",
    )
    parser.add_argument(
        "--agent-timeout-retries",
        type=int,
        default=1,
        help="Retries after timeout for delegated root/chunk agent calls.",
    )
    parser.add_argument(
        "--joint-recall-boost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable aggressive co-location category expansion to improve joint recall.",
    )
    parser.add_argument("--semantic-checks", default="strict", choices=["strict", "off"], help="Semantic faithfulness mode")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-trace checkpoints for interrupted runs",
    )
    parser.add_argument("--notes", default="", help="Free-text notes for the experiment record")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        trail_data_dir=args.trail_data_dir,
        gold_dir=args.gold_dir,
        output_dir=args.output_dir,
        model=args.model,
        root_model=args.root_model,
        chunk_model=args.chunk_model,
        approach=args.approach,
        subset=args.subset,
        prompt_version=args.prompt_version,
        split=args.split,
        max_workers=args.max_workers,
        max_chunks=args.max_chunks,
        max_num_agents=args.max_num_agents,
        agent_call_timeout_seconds=args.agent_call_timeout_seconds,
        agent_timeout_retries=args.agent_timeout_retries,
        joint_recall_boost=args.joint_recall_boost,
        semantic_checks=args.semantic_checks,
        resume=args.resume,
        notes=args.notes,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
