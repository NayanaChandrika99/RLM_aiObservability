# ABOUTME: Validates TRAIL experiment infrastructure including dev subset, prompt v2, and experiment runner.
# ABOUTME: Tests are deterministic and do not call external LLM APIs.

from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path

import pytest


def test_dev_subset_manifest_is_valid_json() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "arcgentica" / "dev_subset_manifest.json"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "subset_id" in manifest
    assert manifest["subset_id"] == "dev18"
    assert "trace_ids" in manifest
    assert isinstance(manifest["trace_ids"], list)
    assert len(manifest["trace_ids"]) == 18
    # All IDs should be 32-char hex strings
    for tid in manifest["trace_ids"]:
        assert isinstance(tid, str)
        assert len(tid) == 32, f"trace_id {tid} is not 32 chars"
    # Should be sorted for determinism
    assert manifest["trace_ids"] == sorted(manifest["trace_ids"])


# ---------------------------------------------------------------------------
# Prompt V2 + smart truncation tests
# ---------------------------------------------------------------------------

from arcgentica.trail_prompt_v2 import (
    TRAIL_SINGLE_PASS_PROMPT_V2,
    build_single_pass_message,
    smart_truncate_trace,
)
from arcgentica.trail_common import TRAIL_LEAF_CATEGORIES


def test_prompt_v2_contains_all_leaf_categories() -> None:
    for category in TRAIL_LEAF_CATEGORIES:
        assert category in TRAIL_SINGLE_PASS_PROMPT_V2, f"Missing category: {category}"


def test_prompt_v2_contains_location_rules() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "Resource Abuse" in prompt
    assert "first" in prompt.lower()
    assert "last" in prompt.lower()
    assert "span_id" in prompt


def test_prompt_v2_contains_detection_hints() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    # Check that detection guidance exists for categories
    assert "DETECT:" in prompt or "detect:" in prompt.lower()


def test_prompt_v2_contains_score_guidance() -> None:
    prompt = TRAIL_SINGLE_PASS_PROMPT_V2
    assert "reliability_score" in prompt
    assert "security_score" in prompt
    assert "instruction_adherence_score" in prompt
    assert "plan_opt_score" in prompt


def test_smart_truncate_preserves_small_trace() -> None:
    trace = {
        "trace_id": "abc123",
        "spans": [
            {
                "span_id": "span_1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": "hello world"}],
                "child_spans": [],
            }
        ],
    }
    result = smart_truncate_trace(trace, max_chars=500_000)
    assert result["trace_id"] == "abc123"
    assert len(result["spans"]) == 1


def test_smart_truncate_truncates_large_trace() -> None:
    spans = []
    for i in range(50):
        spans.append({
            "span_id": f"span_{i:04d}",
            "span_name": f"op_{i}",
            "status_code": "Error" if i == 5 else "Unset",
            "status_message": "some failure" if i == 5 else "",
            "span_attributes": {},
            "logs": [{"body": "x" * 2000}],
            "child_spans": [],
        })
    trace = {"trace_id": "big_trace", "spans": spans}
    result = smart_truncate_trace(trace, max_chars=10_000)
    result_text = json.dumps(result)
    # Should be truncated
    assert len(result_text) <= 20_000  # some overhead
    # Error span should be preserved
    error_spans = [s for s in result["spans"] if s.get("status_code") == "Error"]
    assert len(error_spans) >= 1
    # Span index should be appended
    assert "span_index" in result


def test_smart_truncate_caps_span_index_for_extreme_trace() -> None:
    spans = []
    for i in range(5000):
        spans.append({
            "span_id": f"span_{i:05d}",
            "span_name": "op",
            "status_code": "Unset",
            "status_message": "",
            "span_attributes": {},
            "logs": [],
            "child_spans": [],
        })
    trace = {"trace_id": "extreme_trace", "spans": spans}
    result = smart_truncate_trace(trace, max_chars=40_000)
    result_text = json.dumps(result)
    assert len(result_text) <= 60_000
    assert "span_index" in result
    assert len(result["span_index"]) < 5000


def test_smart_truncate_returns_for_impossible_budget() -> None:
    trace = {
        "trace_id": "tiny_budget_trace",
        "spans": [
            {
                "span_id": "root",
                "span_name": "main",
                "status_code": "Error",
                "status_message": "x" * 2000,
                "span_attributes": {"detail": "y" * 2000},
                "logs": [{"body": "z" * 2000}],
                "child_spans": [],
            }
        ],
    }

    result_holder: dict[str, object] = {}

    def _run() -> None:
        result_holder["result"] = smart_truncate_trace(trace, max_chars=10)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive(), "smart_truncate_trace should return even when max_chars cannot be satisfied."
    assert isinstance(result_holder.get("result"), dict)


def test_build_single_pass_message_returns_string() -> None:
    trace = {
        "trace_id": "test_trace",
        "spans": [
            {
                "span_id": "sp1",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    msg = build_single_pass_message(trace)
    assert isinstance(msg, str)
    assert "sp1" in msg
    assert "errors" in msg.lower()


# ---------------------------------------------------------------------------
# Single-pass mode tests
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from arcgentica.trail_agent import analyze_trace


def _make_trace(trace_id: str = "t1", span_id: str = "s1", log_text: str = "ok") -> dict:
    return {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": span_id,
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [{"body": log_text}],
                "child_spans": [],
            }
        ],
    }


def test_single_pass_mode_calls_litellm() -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "missing format",
                    "description": "Output format wrong",
                    "impact": "LOW",
                }
            ],
            "scores": [
                {
                    "reliability_score": 4,
                    "reliability_reasoning": "Mostly reliable",
                    "security_score": 5,
                    "security_reasoning": "No issues",
                    "instruction_adherence_score": 3,
                    "instruction_adherence_reasoning": "Missed format",
                    "plan_opt_score": 4,
                    "plan_opt_reasoning": "Decent plan",
                    "overall": 4.0,
                }
            ],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response) as mock_comp:
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    mock_comp.assert_called_once()
    assert result["trace_id"] == "t1"
    assert len(result["errors"]) == 1
    assert result["errors"][0]["category"] == "Formatting Errors"
    assert result["errors"][0]["location"] == "s1"
    assert len(result["scores"]) == 1
    assert result["scores"][0]["reliability_score"] == 4


def test_single_pass_mode_falls_back_on_error() -> None:
    with patch("arcgentica.trail_agent.completion", side_effect=Exception("API down")):
        result = analyze_trace(
            _make_trace(log_text="timed out while waiting"),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    assert result["trace_id"] == "t1"
    categories = [e["category"] for e in result["errors"]]
    assert "Timeout Issues" in categories


def test_single_pass_gpt5_mini_does_not_send_temperature() -> None:
    captured_kwargs: dict[str, object] = {}

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    def _fake_completion(**kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        return mock_response

    with patch("arcgentica.trail_agent.completion", side_effect=_fake_completion):
        analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    assert "temperature" not in captured_kwargs
    assert captured_kwargs.get("drop_params") is True
    assert captured_kwargs.get("timeout") == 120


def test_single_pass_fallback_exposes_diagnostics() -> None:
    with patch("arcgentica.trail_agent.completion", side_effect=RuntimeError("API down")):
        result = analyze_trace(
            _make_trace(log_text="timed out while waiting"),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    diagnostics = result.get("analysis_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("analysis_mode") == "heuristic_fallback"
    assert diagnostics.get("fallback_from") == "single_pass"
    assert diagnostics.get("error_type") == "RuntimeError"


# ---------------------------------------------------------------------------
# Experiment runner tests
# ---------------------------------------------------------------------------

from arcgentica.trail_experiment import (
    load_subset_trace_ids,
    ExperimentConfig,
    run_experiment,
    _build_cross_trace_cluster_report,
)
import arcgentica.trail_experiment as trail_experiment


def test_load_subset_trace_ids_dev18() -> None:
    ids = load_subset_trace_ids("dev18")
    assert isinstance(ids, list)
    assert len(ids) == 18
    assert all(isinstance(tid, str) for tid in ids)


def test_load_subset_trace_ids_full_returns_none() -> None:
    ids = load_subset_trace_ids("full")
    assert ids is None


def test_experiment_config_defaults() -> None:
    cfg = ExperimentConfig(
        experiment_id="test_001",
        trail_data_dir=Path("/tmp/data"),
        gold_dir=Path("/tmp/gold"),
        output_dir=Path("/tmp/out"),
    )
    assert cfg.model == "openai/gpt-5-mini"
    assert cfg.approach == "on"
    assert cfg.subset == "dev18"
    assert cfg.prompt_version == "v2"
    assert cfg.split == "GAIA"
    assert cfg.max_workers == 5
    assert cfg.root_model is None
    assert cfg.chunk_model is None


def test_process_trace_file_passes_split_models_to_analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_payload = {
        "trace_id": "trace_split_models",
        "spans": [
            {
                "span_id": "span_0",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_analyze(trace_payload: dict, model: str, **kwargs: object) -> dict:
        captured["trace_id"] = trace_payload.get("trace_id")
        captured["model"] = model
        captured["root_model"] = kwargs.get("root_model")
        captured["chunk_model"] = kwargs.get("chunk_model")
        return {"trace_id": str(trace_payload.get("trace_id", "")), "errors": [], "scores": [{"overall": 5.0}]}

    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", _fake_analyze)

    cfg = ExperimentConfig(
        experiment_id="split_models",
        trail_data_dir=tmp_path,
        gold_dir=tmp_path,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        root_model="openai/gpt-5.2",
        chunk_model="openai/gpt-5-mini",
    )

    trace_result = trail_experiment._process_trace_file(trace_path, cfg)
    assert trace_result["status"] == "ok"
    assert captured["trace_id"] == "trace_split_models"
    assert captured["model"] == "openai/gpt-5-mini"
    assert captured["root_model"] == "openai/gpt-5.2"
    assert captured["chunk_model"] == "openai/gpt-5-mini"


def test_run_experiment_rejects_non_repl_approach(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_id="reject_non_repl",
        trail_data_dir=tmp_path / "data",
        gold_dir=tmp_path / "gold",
        output_dir=tmp_path / "out",
        approach="single_pass",
    )
    with pytest.raises(ValueError, match="REPL-only"):
        run_experiment(cfg)


def test_run_experiment_resumes_without_reanalyzing_completed_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace_ids = ["resume_trace_1", "resume_trace_2"]
    for trace_id in trace_ids:
        trace = {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": f"{trace_id}_span",
                    "span_name": "main",
                    "status_code": "Unset",
                    "status_message": "",
                    "span_attributes": {},
                    "logs": [],
                    "child_spans": [],
                }
            ],
        }
        (data_dir / f"{trace_id}.json").write_text(json.dumps(trace), encoding="utf-8")
        (gold_dir / f"{trace_id}.json").write_text(json.dumps({"errors": [], "scores": [{"overall": 5.0}]}), encoding="utf-8")

    calls = {"count": 0}

    def _fake_analyze(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del model
        del kwargs
        calls["count"] += 1
        return {"trace_id": trace_payload["trace_id"], "errors": [], "scores": [{"overall": 5.0}]}

    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", _fake_analyze)
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    cfg = ExperimentConfig(
        experiment_id="resume_repl_only",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        subset="full",
        split="GAIA",
        approach="on",
        max_workers=1,
    )

    run_experiment(cfg)
    assert calls["count"] == 2

    calls["count"] = 0
    run_experiment(cfg)
    assert calls["count"] == 0

    progress_dir = tmp_path / "out" / "resume_repl_only" / "progress"
    assert progress_dir.exists()
    assert len(list(progress_dir.glob("*.json"))) == 2


def test_preflight_requires_agentica(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_id="preflight_agentica",
        trail_data_dir=tmp_path / "data",
        gold_dir=tmp_path / "gold",
        output_dir=tmp_path / "out",
        approach="on",
    )

    def _fake_import_module(name: str) -> object:
        if name == "agentica":
            raise ModuleNotFoundError("No module named 'agentica'")
        return object()

    monkeypatch.setattr("arcgentica.trail_experiment.importlib.import_module", _fake_import_module)
    with pytest.raises(ModuleNotFoundError, match="requires 'agentica'"):
        trail_experiment._preflight_repl_environment(cfg)


def test_run_experiment_writes_outputs_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run experiment with mocked LLM on 2 synthetic traces."""
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace1 = {
        "trace_id": "aaaa1111bbbb2222cccc3333dddd4444",
        "spans": [{"span_id": "sp1", "span_name": "main", "status_code": "Unset",
                    "status_message": "", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    trace2 = {
        "trace_id": "eeee5555ffff6666aaaa7777bbbb8888",
        "spans": [{"span_id": "sp2", "span_name": "main", "status_code": "Error",
                    "status_message": "timed out", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    (data_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(trace1))
    (data_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(trace2))

    gold1 = {"errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]}
    gold2 = {"errors": [{"category": "Timeout Issues", "location": "sp2",
             "evidence": "timed out", "description": "timeout", "impact": "HIGH"}],
             "scores": [{"reliability_score": 2, "reliability_reasoning": "bad",
             "security_score": 5, "security_reasoning": "ok",
             "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
             "plan_opt_score": 2, "plan_opt_reasoning": "bad", "overall": 3.0}]}
    (gold_dir / "aaaa1111bbbb2222cccc3333dddd4444.json").write_text(json.dumps(gold1))
    (gold_dir / "eeee5555ffff6666aaaa7777bbbb8888.json").write_text(json.dumps(gold2))

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({
        "errors": [], "scores": [{"reliability_score": 5, "reliability_reasoning": "ok",
        "security_score": 5, "security_reasoning": "ok",
        "instruction_adherence_score": 5, "instruction_adherence_reasoning": "ok",
        "plan_opt_score": 5, "plan_opt_reasoning": "ok", "overall": 5.0}]
    })})]

    cfg = ExperimentConfig(
        experiment_id="test_run",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
    )
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = run_experiment(cfg)

    assert result["experiment_id"] == "test_run"
    assert "metrics" in result
    assert "weighted_f1" in result["metrics"]
    assert result["traces_processed"] == 2

    output_dir = tmp_path / "out" / "test_run"
    assert output_dir.exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "weighted_f1" in metrics

    log_path = tmp_path / "out" / "experiment_log.json"
    assert log_path.exists()


def test_run_experiment_avoids_constant_input_correlation_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace1 = {
        "trace_id": "trace_const_1",
        "spans": [{"span_id": "sp1", "span_name": "main", "status_code": "Unset",
                    "status_message": "", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    trace2 = {
        "trace_id": "trace_const_2",
        "spans": [{"span_id": "sp2", "span_name": "main", "status_code": "Unset",
                    "status_message": "", "span_attributes": {}, "logs": [],
                    "child_spans": []}],
    }
    (data_dir / "trace_const_1.json").write_text(json.dumps(trace1), encoding="utf-8")
    (data_dir / "trace_const_2.json").write_text(json.dumps(trace2), encoding="utf-8")

    constant_score = {
        "reliability_score": 5,
        "reliability_reasoning": "ok",
        "security_score": 5,
        "security_reasoning": "ok",
        "instruction_adherence_score": 5,
        "instruction_adherence_reasoning": "ok",
        "plan_opt_score": 5,
        "plan_opt_reasoning": "ok",
        "overall": 5.0,
    }
    gold = {"errors": [], "scores": [constant_score]}
    (gold_dir / "trace_const_1.json").write_text(json.dumps(gold), encoding="utf-8")
    (gold_dir / "trace_const_2.json").write_text(json.dumps(gold), encoding="utf-8")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [constant_score]})})]

    cfg = ExperimentConfig(
        experiment_id="test_const_corr",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
    )
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch("arcgentica.trail_agent.completion", return_value=mock_response):
            run_experiment(cfg)

    warning_names = {w.category.__name__ for w in caught}
    assert "ConstantInputWarning" not in warning_names


def test_run_experiment_uses_threadpool_when_max_workers_gt_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    for idx in range(2):
        trace_id = f"trace_parallel_{idx}"
        trace = {
            "trace_id": trace_id,
            "spans": [{"span_id": f"sp{idx}", "span_name": "main", "status_code": "Unset",
                        "status_message": "", "span_attributes": {}, "logs": [],
                        "child_spans": []}],
        }
        (data_dir / f"{trace_id}.json").write_text(json.dumps(trace), encoding="utf-8")
        (gold_dir / f"{trace_id}.json").write_text(json.dumps({"errors": [], "scores": [{"overall": 5.0}]}), encoding="utf-8")

    calls: dict[str, int] = {"max_workers": 0, "submitted": 0}

    class _FakeFuture:
        def __init__(self, fn: object, *args: object, **kwargs: object) -> None:
            self._exc: Exception | None = None
            self._result: object = None
            try:
                self._result = fn(*args, **kwargs)  # type: ignore[misc]
            except Exception as exc:  # pragma: no cover - defensive
                self._exc = exc

        def result(self) -> object:
            if self._exc is not None:
                raise self._exc
            return self._result

    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            calls["max_workers"] = max_workers

        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type
            del exc
            del tb
            return False

        def submit(self, fn: object, *args: object, **kwargs: object) -> _FakeFuture:
            calls["submitted"] += 1
            return _FakeFuture(fn, *args, **kwargs)

    def _fake_as_completed(futures: object) -> list[object]:
        return list(futures)  # type: ignore[arg-type]

    def _fake_analyze(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {"trace_id": "x", "errors": [], "scores": [{"overall": 5.0}]}

    monkeypatch.setattr("arcgentica.trail_experiment.ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("arcgentica.trail_experiment.as_completed", _fake_as_completed)
    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", _fake_analyze)
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    cfg = ExperimentConfig(
        experiment_id="test_parallel_branch",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        subset="full",
        split="GAIA",
        max_workers=3,
    )
    run_experiment(cfg)

    assert calls["max_workers"] == 3
    assert calls["submitted"] == 2


def test_single_pass_validates_locations() -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message={"content": json.dumps({
            "errors": [
                {
                    "category": "Formatting Errors",
                    "location": "s1",
                    "evidence": "valid location",
                    "description": "ok",
                    "impact": "LOW",
                },
                {
                    "category": "Goal Deviation",
                    "location": "INVALID_SPAN",
                    "evidence": "bad location",
                    "description": "wrong",
                    "impact": "HIGH",
                },
            ],
            "scores": [{"reliability_score": 3, "reliability_reasoning": "ok",
                        "security_score": 5, "security_reasoning": "ok",
                        "instruction_adherence_score": 3, "instruction_adherence_reasoning": "ok",
                        "plan_opt_score": 3, "plan_opt_reasoning": "ok", "overall": 3.5}],
        })})
    ]

    with patch("arcgentica.trail_agent.completion", return_value=mock_response):
        result = analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
        )

    locations = [e["location"] for e in result["errors"]]
    assert all(loc == "s1" for loc in locations), f"Expected all locations to be 's1', got {locations}"


def test_single_pass_prompt_version_changes_prompt() -> None:
    captured_messages: list[str] = []

    def _fake_completion(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            content = messages[0].get("content", "")
            if isinstance(content, str):
                captured_messages.append(content)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]
        return mock_response

    with patch("arcgentica.trail_agent.completion", side_effect=_fake_completion):
        analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
            prompt_version="v1",
        )
        analyze_trace(
            _make_trace(),
            model="openai/gpt-5-mini",
            agentic_mode="single_pass",
            prompt_version="v2",
        )

    assert len(captured_messages) == 2
    assert captured_messages[0] != captured_messages[1]
    assert "detect:" not in captured_messages[0].lower()
    assert "detect:" in captured_messages[1].lower()


def test_single_pass_v2_uses_max_span_text_chars_budget() -> None:
    captured_max_chars: list[int] = []

    def _fake_builder(trace: dict, max_chars: int = 200_000) -> str:
        del trace
        captured_max_chars.append(max_chars)
        return "prompt"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch("arcgentica.trail_agent.build_single_pass_message", side_effect=_fake_builder):
        with patch("arcgentica.trail_agent.completion", return_value=mock_response):
            analyze_trace(
                _make_trace(),
                model="openai/gpt-5-mini",
                agentic_mode="single_pass",
                prompt_version="v2",
                max_span_text_chars=150,
            )

    assert captured_max_chars == [20_000]


def test_single_pass_v2_budget_is_capped_for_large_default() -> None:
    captured_max_chars: list[int] = []

    def _fake_builder(trace: dict, max_chars: int = 200_000) -> str:
        del trace
        captured_max_chars.append(max_chars)
        return "prompt"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch("arcgentica.trail_agent.build_single_pass_message", side_effect=_fake_builder):
        with patch("arcgentica.trail_agent.completion", return_value=mock_response):
            analyze_trace(
                _make_trace(),
                model="openai/gpt-5-mini",
                agentic_mode="single_pass",
                prompt_version="v2",
                max_span_text_chars=1200,
            )

    assert captured_max_chars == [20_000]


def test_single_pass_retries_with_smaller_budget_on_context_window_error() -> None:
    captured_max_chars: list[int] = []

    class ContextWindowExceededError(RuntimeError):
        pass

    def _fake_builder(trace: dict, max_chars: int = 200_000) -> str:
        del trace
        captured_max_chars.append(max_chars)
        return f"prompt-{max_chars}"

    mock_success = MagicMock()
    mock_success.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch("arcgentica.trail_agent.build_single_pass_message", side_effect=_fake_builder):
        with patch(
            "arcgentica.trail_agent.completion",
            side_effect=[ContextWindowExceededError("too many tokens"), mock_success],
        ):
            result = analyze_trace(
                _make_trace(),
                model="openai/gpt-4o-mini",
                agentic_mode="single_pass",
                prompt_version="v2",
                max_span_text_chars=1200,
            )

    assert len(captured_max_chars) >= 2
    assert captured_max_chars[0] == 40_000
    assert captured_max_chars[1] == 20_000
    assert "analysis_diagnostics" not in result


def test_single_pass_retries_on_max_context_length_message() -> None:
    captured_max_chars: list[int] = []

    def _fake_builder(trace: dict, max_chars: int = 200_000) -> str:
        del trace
        captured_max_chars.append(max_chars)
        return f"prompt-{max_chars}"

    mock_success = MagicMock()
    mock_success.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch("arcgentica.trail_agent.build_single_pass_message", side_effect=_fake_builder):
        with patch(
            "arcgentica.trail_agent.completion",
            side_effect=[
                RuntimeError(
                    "This model's maximum context length is 128000 tokens, however you requested 140000 tokens."
                ),
                mock_success,
            ],
        ):
            result = analyze_trace(
                _make_trace(),
                model="openai/gpt-4o-mini",
                agentic_mode="single_pass",
                prompt_version="v2",
                max_span_text_chars=1200,
            )

    assert len(captured_max_chars) >= 2
    assert captured_max_chars[0] == 40_000
    assert captured_max_chars[1] == 20_000
    assert "analysis_diagnostics" not in result


def test_single_pass_retries_rate_limit_error_without_fallback() -> None:
    class RateLimitError(RuntimeError):
        pass

    mock_success = MagicMock()
    mock_success.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch(
        "arcgentica.trail_agent.completion",
        side_effect=[RateLimitError("Please try again in 0.01s"), mock_success],
    ):
        with patch("arcgentica.trail_agent.time.sleep", return_value=None) as sleep_mock:
            result = analyze_trace(
                _make_trace(),
                model="openai/gpt-5-mini",
                agentic_mode="single_pass",
                prompt_version="v2",
            )

    sleep_mock.assert_called_once()
    assert "analysis_diagnostics" not in result


def test_run_experiment_writes_semantic_report_for_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace = {
        "trace_id": "trace_semantic",
        "spans": [
            {
                "span_id": "root_span",
                "span_name": "main",
                "status_code": "Unset",
                "status_message": "",
                "span_attributes": {"detail": "timed out while waiting"},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    (data_dir / "trace_semantic.json").write_text(json.dumps(trace), encoding="utf-8")
    gold = {
        "errors": [
            {
                "category": "Timeout Issues",
                "location": "root_span",
                "evidence": "timed out while waiting",
                "description": "timeout",
                "impact": "HIGH",
            }
        ],
        "scores": [{"overall": 3.0}],
    }
    (gold_dir / "trace_semantic.json").write_text(json.dumps(gold), encoding="utf-8")

    def fake_analyze_trace(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {
            "trace_id": "trace_semantic",
            "errors": [
                {
                    "category": "Timeout Issues",
                    "location": "missing_span",
                    "evidence": "ungrounded evidence text",
                    "description": "bad",
                    "impact": "HIGH",
                }
            ],
            "scores": [{"overall": 1.0}],
        }

    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", fake_analyze_trace)
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    cfg = ExperimentConfig(
        experiment_id="strict_semantic_report",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
        semantic_checks="strict",
    )
    result = run_experiment(cfg)

    semantic_report_path = tmp_path / "out" / "strict_semantic_report" / "semantic_report.json"
    assert semantic_report_path.exists()
    semantic_report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    assert semantic_report["mode"] == "strict"
    assert semantic_report["totals"]["traces_processed"] == 1
    assert semantic_report["totals"]["location_repaired"] == 1
    assert semantic_report["totals"]["evidence_repaired"] == 1
    assert semantic_report["totals"]["dropped_errors"] == 0
    assert result["semantic"]["report_path"] == str(semantic_report_path)


def test_run_experiment_aggregates_delegation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace = {
        "trace_id": "trace_delegation",
        "spans": [
            {
                "span_id": "root_span",
                "span_name": "main",
                "status_code": "Error",
                "status_message": "timed out",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    (data_dir / "trace_delegation.json").write_text(json.dumps(trace), encoding="utf-8")
    gold = {
        "errors": [
            {
                "category": "Timeout Issues",
                "location": "root_span",
                "evidence": "timed out",
                "description": "timeout",
                "impact": "HIGH",
            }
        ],
        "scores": [{"overall": 3.0}],
    }
    (gold_dir / "trace_delegation.json").write_text(json.dumps(gold), encoding="utf-8")

    def fake_analyze_trace(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {
            "trace_id": "trace_delegation",
            "errors": [
                {
                    "category": "Timeout Issues",
                    "location": "root_span",
                    "evidence": "timed out",
                    "description": "timeout",
                    "impact": "HIGH",
                }
            ],
            "scores": [{"overall": 3.0}],
            "analysis_diagnostics": {
                "analysis_mode": "agentic_repl",
                "delegation_failures": 2,
                "delegation_failed_chunk_ids": [1, 2],
            },
        }

    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", fake_analyze_trace)
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    cfg = ExperimentConfig(
        experiment_id="delegation_failure_totals",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        model="openai/gpt-5-mini",
        subset="full",
        split="GAIA",
        semantic_checks="strict",
    )
    run_experiment(cfg)

    semantic_report_path = tmp_path / "out" / "delegation_failure_totals" / "semantic_report.json"
    semantic_report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    assert semantic_report["totals"]["delegation_failures"] == 2


# ---------------------------------------------------------------------------
# Phase 2a: recursive child_spans pruning
# ---------------------------------------------------------------------------


def test_smart_truncate_prunes_nested_child_spans() -> None:
    """Phase 2a must recursively prune child_spans, not just top-level spans.

    TRAIL traces often have 1 root span with many nested children.
    Before this fix, Phase 2 was a no-op for such traces.
    """
    from arcgentica.trail_prompt_v2 import smart_truncate_trace

    # Build a trace with 1 root span containing 50 nested children
    children = []
    for i in range(50):
        children.append({
            "span_id": f"child_{i:03d}",
            "span_name": f"step_{i}",
            "status_code": "OK",
            "status_message": "",
            "span_attributes": {"detail": "x" * 2000},
            "logs": [{"body": "y" * 2000}],
            "child_spans": [],
        })
    # Mark one child as error so it's kept
    children[10]["status_code"] = "ERROR"
    children[10]["status_message"] = "something failed"

    trace = {
        "trace_id": "nested_trace",
        "spans": [{
            "span_id": "root",
            "span_name": "root_span",
            "status_code": "OK",
            "status_message": "",
            "span_attributes": {},
            "logs": [],
            "child_spans": children,
        }],
    }

    # Without fix, this trace would be too large; with fix, child spans get pruned
    result = smart_truncate_trace(trace, max_chars=30_000)
    serialized = json.dumps(result)
    assert len(serialized) <= 30_000

    # The span_index should still list all original span_ids
    span_ids_in_index = {e["span_id"] for e in result.get("span_index", [])}
    assert "root" in span_ids_in_index
    assert "child_010" in span_ids_in_index  # Error span should be in index

    # The error child and first/last children should be preserved in the actual tree
    root_span = result["spans"][0]
    remaining_ids = {c["span_id"] for c in root_span.get("child_spans", [])}
    assert "child_000" in remaining_ids, "First child should be kept"
    assert "child_049" in remaining_ids, "Last child should be kept"
    assert "child_010" in remaining_ids, "Error child should be kept"
    # Some middle children should have been dropped
    assert len(remaining_ids) < 50


def test_smart_truncate_prunes_deeply_nested_spans() -> None:
    """Phase 2a should handle multi-level nesting (depth > 2)."""
    from arcgentica.trail_prompt_v2 import smart_truncate_trace

    # Build depth-3 trace: root -> 3 children -> 20 grandchildren each
    grandchildren_per_child = 20
    children = []
    for i in range(3):
        gchildren = []
        for j in range(grandchildren_per_child):
            gchildren.append({
                "span_id": f"gc_{i}_{j:02d}",
                "span_name": f"grandchild_{i}_{j}",
                "status_code": "OK",
                "status_message": "",
                "span_attributes": {"data": "z" * 1500},
                "logs": [],
                "child_spans": [],
            })
        children.append({
            "span_id": f"child_{i}",
            "span_name": f"child_{i}",
            "status_code": "OK",
            "status_message": "",
            "span_attributes": {},
            "logs": [],
            "child_spans": gchildren,
        })

    trace = {
        "trace_id": "deep_nested",
        "spans": [{
            "span_id": "root",
            "span_name": "root",
            "status_code": "OK",
            "status_message": "",
            "span_attributes": {},
            "logs": [],
            "child_spans": children,
        }],
    }

    result = smart_truncate_trace(trace, max_chars=25_000)
    assert len(json.dumps(result)) <= 25_000

    # Root and top-level children should still exist
    assert len(result["spans"]) == 1
    root = result["spans"][0]
    assert root["span_id"] == "root"
    # Some grandchildren should have been pruned
    total_gc = sum(
        len(c.get("child_spans", []))
        for c in root.get("child_spans", [])
    )
    assert total_gc < 3 * grandchildren_per_child


# ---------------------------------------------------------------------------
# Token preflight
# ---------------------------------------------------------------------------


def test_single_pass_skips_budget_when_token_preflight_fails() -> None:
    """If estimated tokens exceed model limit, budget should be skipped
    without calling the API (no context window error needed)."""
    from arcgentica.trail_agent import _estimate_prompt_tokens, _model_input_token_limit

    # Verify helpers work
    assert _estimate_prompt_tokens("a" * 400) == 100
    assert _model_input_token_limit("openai/gpt-5-mini") == 128_000
    assert _model_input_token_limit("openai/gpt-5.2") == 256_000

    captured_prompts: list[str] = []

    def _fake_builder(trace: dict, max_chars: int = 200_000) -> str:
        del trace
        # Return a prompt that's ~600K chars (150K tokens) for large budgets
        # and ~200K chars (50K tokens) for small budgets
        if max_chars >= 30_000:
            prompt = "x" * 600_000
        else:
            prompt = "x" * 200_000
        captured_prompts.append(f"budget={max_chars},len={len(prompt)}")
        return prompt

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message={"content": json.dumps({"errors": [], "scores": [{"overall": 5.0}]})})]

    with patch("arcgentica.trail_agent.build_single_pass_message", side_effect=_fake_builder):
        with patch("arcgentica.trail_agent.completion", return_value=mock_response) as comp_mock:
            result = analyze_trace(
                _make_trace(),
                model="openai/gpt-4o-mini",  # 128K limit
                agentic_mode="single_pass",
                prompt_version="v2",
                max_span_text_chars=1200,
            )

    # The large-budget prompt (150K tokens) should be skipped by preflight
    # Only the small-budget prompt (50K tokens) should reach completion()
    assert comp_mock.call_count == 1
    assert "analysis_diagnostics" not in result


def test_run_experiment_records_cross_trace_cluster_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data" / "GAIA"
    data_dir.mkdir(parents=True)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)

    trace = {
        "trace_id": "trace_cluster_1",
        "spans": [
            {
                "span_id": "span_cluster_1",
                "span_name": "main",
                "status_code": "Error",
                "status_message": "tool schema mismatch",
                "span_attributes": {},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    (data_dir / "trace_cluster_1.json").write_text(json.dumps(trace), encoding="utf-8")
    (gold_dir / "trace_cluster_1.json").write_text(
        json.dumps({"errors": [], "scores": [{"overall": 3.0}]}),
        encoding="utf-8",
    )

    def _fake_analyze(trace_payload: dict, model: str, **kwargs: object) -> dict:
        del trace_payload
        del model
        del kwargs
        return {
            "trace_id": "trace_cluster_1",
            "errors": [
                {
                    "category": "Tool Definition Issues",
                    "location": "span_cluster_1",
                    "evidence": "unexpected keyword argument step_count",
                    "description": "tool schema mismatch in page navigation",
                    "impact": "MEDIUM",
                }
            ],
            "scores": [{"overall": 3.0}],
        }

    monkeypatch.setattr("arcgentica.trail_experiment.analyze_trace", _fake_analyze)
    monkeypatch.setattr("arcgentica.trail_experiment._preflight_repl_environment", lambda config: None)

    cfg = ExperimentConfig(
        experiment_id="cluster_artifact_run",
        trail_data_dir=tmp_path / "data",
        gold_dir=gold_dir,
        output_dir=tmp_path / "out",
        subset="full",
        split="GAIA",
        approach="on",
        max_workers=1,
    )
    result = run_experiment(cfg)

    cluster_path = tmp_path / "out" / "cluster_artifact_run" / "cross_trace_clusters.json"
    assert cluster_path.exists()
    cluster_report = json.loads(cluster_path.read_text(encoding="utf-8"))
    assert cluster_report["total_errors"] == 1
    assert cluster_report["cluster_count"] == 1
    assert result["cross_trace_clusters"]["report_path"] == str(cluster_path)
    assert result["cross_trace_clusters"]["cluster_count"] == 1


def test_build_cross_trace_cluster_report_groups_related_errors(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir(parents=True)

    (generated_dir / "trace_a.json").write_text(
        json.dumps(
            {
                "trace_id": "trace_a",
                "errors": [
                    {
                        "category": "Tool Definition Issues",
                        "location": "span_a",
                        "evidence": "PageDownTool unexpected keyword argument step_count",
                        "description": "tool schema mismatch on page navigation",
                        "impact": "MEDIUM",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (generated_dir / "trace_b.json").write_text(
        json.dumps(
            {
                "trace_id": "trace_b",
                "errors": [
                    {
                        "category": "Tool Definition Issues",
                        "location": "span_b",
                        "evidence": "PageDownTool got unexpected keyword argument",
                        "description": "schema mismatch while paging",
                        "impact": "MEDIUM",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (generated_dir / "trace_c.json").write_text(
        json.dumps(
            {
                "trace_id": "trace_c",
                "errors": [
                    {
                        "category": "Resource Not Found",
                        "location": "span_c",
                        "evidence": "VisitTool returned 404 not found",
                        "description": "resource missing from target path",
                        "impact": "MEDIUM",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = _build_cross_trace_cluster_report(generated_dir)

    assert report["total_errors"] == 3
    assert report["cluster_count"] == 2
    assert len(report["clusters"]) == 2

    largest = report["clusters"][0]
    assert largest["size"] == 2
    assert largest["dominant_category"] == "Tool Definition Issues"
    assert set(largest["trace_ids"]) == {"trace_a", "trace_b"}
