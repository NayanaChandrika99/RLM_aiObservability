# ABOUTME: Implements deterministic incident representative selection plus optional per-trace LLM synthesis.
# ABOUTME: Produces contract-aligned incident dossiers with evidence-linked hypotheses and actions.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from investigator.inspection_api import PhoenixInspectionAPI
from investigator.runtime.llm_client import (
    ModelOutputInvalidError,
    OpenAIModelClient,
    RuntimeModelClient,
    StructuredGenerationRequest,
    StructuredGenerationUsage,
)
from investigator.runtime.llm_loop import run_structured_generation_loop
from investigator.runtime.prompt_registry import PromptDefinition, get_prompt_definition
from investigator.runtime.recursive_loop import RecursiveLoop, RecursiveLoopResult
from investigator.runtime.recursive_planner import StructuredActionPlanner
from investigator.runtime.sandbox import SandboxGuard
from investigator.runtime.tool_registry import ToolRegistry
from investigator.runtime.contracts import (
    EvidenceRef,
    IncidentDossier,
    IncidentHypothesis,
    IncidentTimelineEvent,
    InputRef,
    RecommendedAction,
    RepresentativeTrace,
    RuntimeBudget,
    SuspectedChange,
    TimeWindow,
    hash_excerpt,
)


@dataclass
class IncidentDossierRequest:
    project_name: str
    time_window_start: str
    time_window_end: str
    filter_expr: str | None = None
    trace_ids_override: list[str] = field(default_factory=list)


_INCIDENT_PROMPT_ID = "incident_dossier_judgment_v1"
_INCIDENT_PROMPT = get_prompt_definition(_INCIDENT_PROMPT_ID)
_RECURSIVE_INCIDENT_PROMPT_ID = "recursive_runtime_action_v1"
_ALLOWED_ACTION_PRIORITIES = {"P0", "P1", "P2"}
_ALLOWED_ACTION_TYPES = {"mitigation", "follow_up_fix"}


class IncidentDossierEngine:
    engine_type = "incident_dossier"
    output_contract_name = "IncidentDossier"
    engine_version = "0.3.0"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = _INCIDENT_PROMPT.prompt_template_hash
    temperature = 0.0

    def __init__(
        self,
        inspection_api: Any | None = None,
        model_client: RuntimeModelClient | None = None,
        *,
        max_representatives: int = 12,
        error_quota: int = 5,
        latency_quota: int = 5,
        cluster_quota: int = 2,
        max_prompt_timeline_events: int = 6,
        use_llm_judgment: bool = False,
        use_recursive_runtime: bool = False,
        recursive_budget: RuntimeBudget | None = None,
        fallback_on_llm_error: bool = False,
    ) -> None:
        self._inspection_api = inspection_api
        self._model_client = model_client
        self._max_representatives = max_representatives
        self._error_quota = error_quota
        self._latency_quota = latency_quota
        self._cluster_quota = cluster_quota
        self._max_prompt_timeline_events = max_prompt_timeline_events
        self._use_llm_judgment = use_llm_judgment
        self._use_recursive_runtime = use_recursive_runtime
        self._recursive_budget = recursive_budget or RuntimeBudget(
            max_iterations=50,
            max_depth=2,
            max_tool_calls=220,
            max_subcalls=90,
            max_tokens_total=320000,
        )
        self._fallback_on_llm_error = fallback_on_llm_error
        self._prompt_definition: PromptDefinition = _INCIDENT_PROMPT
        if self._use_llm_judgment and self._use_recursive_runtime:
            self._prompt_definition = get_prompt_definition(_RECURSIVE_INCIDENT_PROMPT_ID)
        self.prompt_template_hash = self._prompt_definition.prompt_template_hash
        model_provider = str(getattr(self._model_client, "model_provider", "") or "").strip()
        if model_provider:
            self.model_provider = model_provider
        self._runtime_signals: dict[str, object] = {}
        self._reset_runtime_signals()

    def _reset_runtime_signals(self) -> None:
        self._runtime_signals = {
            "iterations": 1,
            "depth_reached": 0,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "model_provider": self.model_provider,
            "incident_judgment_mode": "deterministic",
            "runtime_state": "completed",
            "budget_reason": "",
            "state_trajectory": [],
            "subcall_metadata": [],
            "subcalls": 0,
        }

    def get_runtime_signals(self) -> dict[str, object]:
        return dict(self._runtime_signals)

    def build_input_ref(self, request: IncidentDossierRequest) -> InputRef:
        return InputRef(
            project_name=request.project_name,
            trace_ids=request.trace_ids_override,
            time_window=TimeWindow(start=request.time_window_start, end=request.time_window_end),
            filter_expr=request.filter_expr,
            controls_version=None,
        )

    def _resolve_inspection_api(self, request: IncidentDossierRequest) -> Any:
        if self._inspection_api is not None:
            return self._inspection_api
        return PhoenixInspectionAPI(project_name=request.project_name)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _coerce_confidence(cls, value: Any, *, default: float) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        return max(0.0, min(1.0, float(default)))

    @staticmethod
    def _coerce_action_priority(value: Any) -> str:
        normalized = str(value or "").strip().upper()
        if normalized in _ALLOWED_ACTION_PRIORITIES:
            return normalized
        return "P1"

    @staticmethod
    def _coerce_action_type(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in _ALLOWED_ACTION_TYPES:
            return normalized
        return "follow_up_fix"

    @staticmethod
    def _dedupe_evidence_refs(evidence_refs: list[EvidenceRef]) -> list[EvidenceRef]:
        deduped: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for evidence in evidence_refs:
            key = (evidence.kind, evidence.ref)
            if key in seen:
                continue
            deduped.append(evidence)
            seen.add(key)
        return deduped

    @staticmethod
    def _signature_from_spans(spans: list[dict[str, Any]]) -> tuple[str, str, str]:
        ordered = sorted(
            spans,
            key=lambda span: (
                str(span.get("start_time") or ""),
                str(span.get("span_id") or ""),
            ),
        )
        root = next(
            (
                span
                for span in ordered
                if str(span.get("parent_id") or "") == ""
            ),
            ordered[0] if ordered else {},
        )
        service = str(root.get("name") or "unknown_service")
        tool_name = "none"
        for span in ordered:
            if str(span.get("span_kind") or "") == "TOOL":
                tool_name = str(span.get("name") or "unknown_tool")
                break
        error_type = "none"
        for span in ordered:
            if str(span.get("status_code") or "") == "ERROR":
                error_type = str(span.get("status_message") or "error").strip().lower() or "error"
                break
        return (service, tool_name, error_type)

    @staticmethod
    def _root_span_id(spans: list[dict[str, Any]]) -> str:
        if not spans:
            return "root-span"
        ordered = sorted(
            spans,
            key=lambda span: (
                str(span.get("start_time") or ""),
                str(span.get("span_id") or ""),
            ),
        )
        for span in ordered:
            if str(span.get("parent_id") or "") == "":
                return str(span.get("span_id") or "root-span")
        return str(ordered[0].get("span_id") or "root-span")

    @staticmethod
    def _evidence_from_profile(profile: dict[str, Any]) -> EvidenceRef:
        root_span_id = str(profile.get("root_span_id") or "root-span")
        trace_id = str(profile.get("trace_id") or "trace-stub")
        summary_text = (
            f"bucket={profile.get('bucket') or ''};"
            f"error_spans={profile.get('error_spans') or 0};"
            f"latency_ms={profile.get('latency_ms') or 0}"
        )
        return EvidenceRef(
            trace_id=trace_id,
            span_id=root_span_id,
            kind="SPAN",
            ref=root_span_id,
            excerpt_hash=hash_excerpt(summary_text),
            ts=profile.get("start_time"),
        )

    @staticmethod
    def _evidence_dict(evidence: EvidenceRef) -> dict[str, Any]:
        return {
            "trace_id": evidence.trace_id,
            "span_id": evidence.span_id,
            "kind": evidence.kind,
            "ref": evidence.ref,
            "excerpt_hash": evidence.excerpt_hash,
            "ts": evidence.ts,
        }

    @staticmethod
    def _change_type_from_paths(paths: list[str]) -> str:
        normalized = [path.lower() for path in paths]
        if any("prompt" in path for path in normalized):
            return "prompt"
        if any("requirements" in path or "pyproject" in path or "poetry.lock" in path for path in normalized):
            return "dependency"
        return "config"

    def _build_suspected_change(
        self,
        *,
        inspection_api: Any,
        request: IncidentDossierRequest,
        lead_evidence: EvidenceRef,
    ) -> tuple[SuspectedChange, IncidentTimelineEvent | None, list[str]]:
        gaps: list[str] = []
        try:
            snapshots = inspection_api.list_config_snapshots(
                request.project_name,
                end_time=request.time_window_end,
            ) or []
        except Exception as exc:
            gaps.append(f"Failed to list config snapshots for incident window: {exc}")
            snapshots = []
        ordered_snapshots = sorted(
            snapshots,
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("snapshot_id") or ""),
            ),
        )
        if len(ordered_snapshots) < 2:
            gaps.append("Insufficient config snapshots to compute diff evidence.")
            return (
                SuspectedChange(
                    change_type="unknown",
                    change_ref="not-identified",
                    diff_ref=None,
                    summary="Config diff correlation unavailable: fewer than two snapshots were found.",
                    evidence_refs=[lead_evidence],
                ),
                None,
                gaps,
            )
        base_snapshot_id = str(ordered_snapshots[-2].get("snapshot_id") or "")
        target_snapshot_id = str(ordered_snapshots[-1].get("snapshot_id") or "")
        if not base_snapshot_id or not target_snapshot_id:
            gaps.append("Config snapshot IDs were missing; cannot compute config diff evidence.")
            return (
                SuspectedChange(
                    change_type="unknown",
                    change_ref="not-identified",
                    diff_ref=None,
                    summary="Config diff correlation unavailable: snapshot IDs were incomplete.",
                    evidence_refs=[lead_evidence],
                ),
                None,
                gaps,
            )
        try:
            diff = inspection_api.get_config_diff(base_snapshot_id, target_snapshot_id)
        except Exception as exc:
            gaps.append(
                f"Failed to compute config diff for snapshots {base_snapshot_id}->{target_snapshot_id}: {exc}"
            )
            return (
                SuspectedChange(
                    change_type="unknown",
                    change_ref=f"{base_snapshot_id}->{target_snapshot_id}",
                    diff_ref=None,
                    summary="Config diff correlation unavailable: diff computation failed.",
                    evidence_refs=[lead_evidence],
                ),
                None,
                gaps,
            )
        diff_ref = str(diff.get("diff_ref") or diff.get("artifact_id") or "")
        changed_paths = [str(path) for path in (diff.get("paths") or []) if str(path)]
        if not diff_ref:
            gaps.append(
                f"Config diff for snapshots {base_snapshot_id}->{target_snapshot_id} missing diff_ref."
            )
            return (
                SuspectedChange(
                    change_type="unknown",
                    change_ref=f"{base_snapshot_id}->{target_snapshot_id}",
                    diff_ref=None,
                    summary="Config diff correlation unavailable: diff reference was missing.",
                    evidence_refs=[lead_evidence],
                ),
                None,
                gaps,
            )
        if not changed_paths:
            return (
                SuspectedChange(
                    change_type="unknown",
                    change_ref=f"{base_snapshot_id}->{target_snapshot_id}",
                    diff_ref=diff_ref,
                    summary=(
                        str(diff.get("summary") or "")
                        or "No changed configuration paths detected in latest snapshot pair."
                    ),
                    evidence_refs=[
                        lead_evidence,
                        EvidenceRef(
                            trace_id=lead_evidence.trace_id,
                            span_id=lead_evidence.span_id,
                            kind="CONFIG_DIFF",
                            ref=diff_ref,
                            excerpt_hash=hash_excerpt(
                                f"{base_snapshot_id}->{target_snapshot_id}:{diff_ref}:no_paths"
                            ),
                            ts=request.time_window_end,
                        ),
                    ],
                ),
                None,
                gaps,
            )
        change_type = self._change_type_from_paths(changed_paths)
        change_ref = str(diff.get("git_commit_target") or f"{base_snapshot_id}->{target_snapshot_id}")
        change_evidence = EvidenceRef(
            trace_id=lead_evidence.trace_id,
            span_id=lead_evidence.span_id,
            kind="CONFIG_DIFF",
            ref=diff_ref,
            excerpt_hash=hash_excerpt(
                f"{base_snapshot_id}->{target_snapshot_id}:{'|'.join(changed_paths)}:{diff_ref}"
            ),
            ts=request.time_window_end,
        )
        summary_prefix = str(diff.get("summary") or f"{len(changed_paths)} changed file(s)")
        path_preview = ", ".join(changed_paths[:3])
        suspected_change = SuspectedChange(
            change_type=change_type,
            change_ref=change_ref,
            diff_ref=diff_ref,
            summary=(
                f"Correlated config diff {base_snapshot_id}->{target_snapshot_id}: "
                f"{summary_prefix}; paths={path_preview}"
            ),
            evidence_refs=[lead_evidence, change_evidence],
        )
        timeline_event = IncidentTimelineEvent(
            timestamp=request.time_window_end,
            event=(
                f"Correlated config diff {base_snapshot_id}->{target_snapshot_id} "
                f"with {len(changed_paths)} changed path(s)."
            ),
            evidence_refs=[lead_evidence, change_evidence],
        )
        return suspected_change, timeline_event, gaps

    def _candidate_from_trace(
        self,
        trace_summary: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trace_id = str(trace_summary.get("trace_id") or "")
        error_spans = sum(1 for span in spans if str(span.get("status_code") or "") == "ERROR")
        signature = self._signature_from_spans(spans)
        latency_ms = self._safe_float(trace_summary.get("latency_ms"))
        return {
            "trace_id": trace_id,
            "latency_ms": latency_ms,
            "error_spans": error_spans,
            "signature": signature,
            "spans": spans,
            "root_span_id": self._root_span_id(spans),
            "start_time": trace_summary.get("start_time"),
            "end_time": trace_summary.get("end_time"),
        }

    @staticmethod
    def _sort_error_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -int(candidate.get("error_spans") or 0),
                -float(candidate.get("latency_ms") or 0.0),
                str(candidate.get("trace_id") or ""),
            ),
        )

    @staticmethod
    def _sort_latency_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.get("latency_ms") or 0.0),
                -int(candidate.get("error_spans") or 0),
                str(candidate.get("trace_id") or ""),
            ),
        )

    def _select_bucket(
        self,
        *,
        candidates: list[dict[str, Any]],
        quota: int,
        bucket_name: str,
        selected_trace_ids: set[str],
        selected_signatures: set[tuple[str, str, str]],
        predicate: Any = None,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(selected) >= quota:
                break
            trace_id = str(candidate.get("trace_id") or "")
            signature = candidate.get("signature")
            if not trace_id or trace_id in selected_trace_ids:
                continue
            if isinstance(signature, tuple) and signature in selected_signatures:
                continue
            if predicate is not None and not predicate(candidate):
                continue
            profile = dict(candidate)
            profile["bucket"] = bucket_name
            if bucket_name == "error":
                profile["bucket_score"] = float(candidate.get("error_spans") or 0.0)
            elif bucket_name == "latency":
                profile["bucket_score"] = float(candidate.get("latency_ms") or 0.0)
            else:
                profile["bucket_score"] = float(candidate.get("latency_ms") or 0.0)
            selected.append(profile)
            selected_trace_ids.add(trace_id)
            if isinstance(signature, tuple):
                selected_signatures.add(signature)
        return selected

    def _choose_representative_profiles(
        self,
        candidates: list[dict[str, Any]],
        *,
        override_trace_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_trace_ids: set[str] = set()
        selected_signatures: set[tuple[str, str, str]] = set()
        error_candidates = self._sort_error_candidates(candidates)
        latency_candidates = self._sort_latency_candidates(candidates)

        selected_profiles: list[dict[str, Any]] = []
        selected_profiles.extend(
            self._select_bucket(
                candidates=error_candidates,
                quota=self._error_quota,
                bucket_name="error",
                selected_trace_ids=selected_trace_ids,
                selected_signatures=selected_signatures,
                predicate=lambda item: int(item.get("error_spans") or 0) > 0,
            )
        )
        selected_profiles.extend(
            self._select_bucket(
                candidates=latency_candidates,
                quota=self._latency_quota,
                bucket_name="latency",
                selected_trace_ids=selected_trace_ids,
                selected_signatures=selected_signatures,
            )
        )
        selected_profiles.extend(
            self._select_bucket(
                candidates=latency_candidates,
                quota=self._cluster_quota,
                bucket_name="cluster",
                selected_trace_ids=selected_trace_ids,
                selected_signatures=selected_signatures,
            )
        )

        if override_trace_ids:
            by_trace_id = {str(candidate.get("trace_id") or ""): candidate for candidate in candidates}
            for trace_id in override_trace_ids:
                candidate = by_trace_id.get(trace_id)
                if candidate is None or trace_id in selected_trace_ids:
                    continue
                profile = dict(candidate)
                profile["bucket"] = "override"
                profile["bucket_score"] = float(candidate.get("latency_ms") or 0.0)
                selected_profiles.append(profile)
                selected_trace_ids.add(trace_id)
                signature = candidate.get("signature")
                if isinstance(signature, tuple):
                    selected_signatures.add(signature)

        priority = {"error": 0, "latency": 1, "cluster": 2, "override": 3}
        ordered = sorted(
            selected_profiles,
            key=lambda profile: (
                priority.get(str(profile.get("bucket") or "cluster"), 99),
                -float(profile.get("bucket_score") or 0.0),
                str(profile.get("trace_id") or ""),
            ),
        )
        return ordered[: self._max_representatives]

    def _resolve_model_client(self) -> RuntimeModelClient:
        if self._model_client is None:
            self._model_client = OpenAIModelClient()
        return self._model_client

    def _llm_trace_synthesis(
        self,
        *,
        request: IncidentDossierRequest,
        profile: dict[str, Any],
        representative_trace: RepresentativeTrace,
        timeline: list[IncidentTimelineEvent],
        suspected_change: SuspectedChange,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str], float, int, StructuredGenerationUsage, str]:
        timeline_snapshot = [
            {
                "timestamp": item.timestamp,
                "event": item.event,
                "evidence_refs": [self._evidence_dict(ref) for ref in item.evidence_refs[:3]],
            }
            for item in sorted(timeline, key=lambda row: row.timestamp)[: self._max_prompt_timeline_events]
        ]
        prompt_payload = {
            "project_name": request.project_name,
            "incident_window": {
                "start": request.time_window_start,
                "end": request.time_window_end,
            },
            "representative_trace": {
                "trace_id": representative_trace.trace_id,
                "why_selected": representative_trace.why_selected,
                "profile": {
                    "bucket": str(profile.get("bucket") or ""),
                    "error_spans": int(profile.get("error_spans") or 0),
                    "latency_ms": float(profile.get("latency_ms") or 0.0),
                    "signature": profile.get("signature"),
                },
                "evidence_refs": [
                    self._evidence_dict(ref) for ref in representative_trace.evidence_refs[:5]
                ],
            },
            "suspected_change": {
                "change_type": suspected_change.change_type,
                "change_ref": suspected_change.change_ref,
                "diff_ref": suspected_change.diff_ref,
                "summary": suspected_change.summary,
                "evidence_refs": [self._evidence_dict(ref) for ref in suspected_change.evidence_refs[:3]],
            },
            "timeline": timeline_snapshot,
        }
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        request_payload = StructuredGenerationRequest(
            model_provider=model_provider,
            model_name=self.model_name,
            temperature=self.temperature,
            system_prompt=self._prompt_definition.prompt_text,
            user_prompt=json.dumps(prompt_payload, sort_keys=True),
            response_schema_name=self._prompt_definition.prompt_id,
            response_schema=self._prompt_definition.response_schema,
        )
        loop_result = run_structured_generation_loop(
            client=model_client,
            request=request_payload,
        )
        payload = loop_result.output
        summary = str(payload.get("incident_summary") or "").strip()
        hypotheses_payload = payload.get("hypotheses")
        actions_payload = payload.get("recommended_actions")
        gaps_payload = payload.get("gaps")
        confidence = self._coerce_confidence(payload.get("confidence"), default=0.6)
        hypotheses = hypotheses_payload if isinstance(hypotheses_payload, list) else []
        actions = actions_payload if isinstance(actions_payload, list) else []
        llm_gaps = [str(item).strip() for item in gaps_payload or [] if str(item).strip()]
        return (
            summary,
            [item for item in hypotheses if isinstance(item, dict)],
            [item for item in actions if isinstance(item, dict)],
            llm_gaps,
            confidence,
            int(loop_result.attempt_count),
            loop_result.usage,
            model_provider,
        )

    def _update_runtime_signals(
        self,
        *,
        usage_total: StructuredGenerationUsage,
        runtime_tracker: dict[str, Any],
    ) -> None:
        llm_used = bool(runtime_tracker.get("llm_used"))
        fallback_used = bool(runtime_tracker.get("fallback_used"))
        recursive_used = bool(runtime_tracker.get("recursive_used"))
        if fallback_used:
            mode = "deterministic_fallback"
        elif recursive_used:
            mode = "recursive_llm"
        elif llm_used:
            mode = "llm"
        else:
            mode = "deterministic"
        self._runtime_signals["iterations"] = max(1, int(runtime_tracker.get("attempts") or 1))
        self._runtime_signals["depth_reached"] = int(runtime_tracker.get("depth_reached") or 0)
        if recursive_used:
            self._runtime_signals["tool_calls"] = int(runtime_tracker.get("tool_calls") or 0)
        else:
            self._runtime_signals["tool_calls"] = int(runtime_tracker.get("llm_calls") or 0)
        self._runtime_signals["tokens_in"] = int(usage_total.tokens_in)
        self._runtime_signals["tokens_out"] = int(usage_total.tokens_out)
        self._runtime_signals["cost_usd"] = float(usage_total.cost_usd)
        self._runtime_signals["model_provider"] = str(
            runtime_tracker.get("model_provider") or self.model_provider
        )
        self._runtime_signals["incident_judgment_mode"] = mode
        self._runtime_signals["runtime_state"] = str(runtime_tracker.get("runtime_state") or "completed")
        self._runtime_signals["budget_reason"] = str(runtime_tracker.get("budget_reason") or "")
        raw_trajectory = runtime_tracker.get("state_trajectory") or []
        self._runtime_signals["state_trajectory"] = [
            str(item) for item in raw_trajectory if str(item).strip()
        ]
        raw_subcall_metadata = runtime_tracker.get("subcall_metadata") or []
        subcall_metadata = [item for item in raw_subcall_metadata if isinstance(item, dict)]
        self._runtime_signals["subcall_metadata"] = subcall_metadata
        self._runtime_signals["subcalls"] = len(subcall_metadata)
        sandbox_violations = runtime_tracker.get("sandbox_violations") or []
        if sandbox_violations:
            self._runtime_signals["sandbox_violations"] = [
                str(item) for item in sandbox_violations if str(item).strip()
            ]

    @staticmethod
    def _merge_recursive_loop_result(
        runtime_tracker: dict[str, Any],
        loop_result: RecursiveLoopResult,
    ) -> None:
        runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + max(
            1, int(loop_result.usage.iterations)
        )
        runtime_tracker["tool_calls"] = int(runtime_tracker.get("tool_calls") or 0) + int(
            loop_result.usage.tool_calls
        )
        runtime_tracker["depth_reached"] = max(
            int(runtime_tracker.get("depth_reached") or 0),
            int(loop_result.usage.depth_reached),
        )
        runtime_tracker.setdefault("state_trajectory", []).extend(loop_result.state_trajectory)
        runtime_tracker.setdefault("subcall_metadata", []).extend(loop_result.subcall_metadata)
        if loop_result.status == "terminated_budget":
            runtime_tracker["runtime_state"] = "terminated_budget"
            if not runtime_tracker.get("budget_reason"):
                runtime_tracker["budget_reason"] = str(loop_result.budget_reason or "")
        if loop_result.status == "failed":
            runtime_tracker["runtime_state"] = "failed"
            if loop_result.error_code == "SANDBOX_VIOLATION":
                runtime_tracker.setdefault("sandbox_violations", []).append(
                    str(loop_result.error_message or "")
                )

    def _recursive_trace_synthesis(
        self,
        *,
        request: IncidentDossierRequest,
        profile: dict[str, Any],
        representative_trace: RepresentativeTrace,
        timeline: list[IncidentTimelineEvent],
        suspected_change: SuspectedChange,
        runtime_tracker: dict[str, Any],
        usage_total: StructuredGenerationUsage,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str], float]:
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        planner = StructuredActionPlanner(
            client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt_id=_RECURSIVE_INCIDENT_PROMPT_ID,
        )
        self.prompt_template_hash = planner.prompt_template_hash
        runtime_tracker["recursive_used"] = True
        runtime_tracker["llm_used"] = True
        runtime_tracker["model_provider"] = model_provider

        timeline_snapshot = [
            {
                "timestamp": item.timestamp,
                "event": item.event,
                "evidence_refs": [self._evidence_dict(ref) for ref in item.evidence_refs[:3]],
            }
            for item in sorted(timeline, key=lambda row: row.timestamp)[: self._max_prompt_timeline_events]
        ]
        planner_seed = {
            "engine_type": self.engine_type,
            "incident_scope": "per_trace_drilldown",
            "project_name": request.project_name,
            "incident_window": {
                "start": request.time_window_start,
                "end": request.time_window_end,
            },
            "representative_trace": {
                "trace_id": representative_trace.trace_id,
                "why_selected": representative_trace.why_selected,
                "profile": {
                    "bucket": str(profile.get("bucket") or ""),
                    "error_spans": int(profile.get("error_spans") or 0),
                    "latency_ms": float(profile.get("latency_ms") or 0.0),
                    "signature": profile.get("signature"),
                },
                "evidence_refs": [
                    self._evidence_dict(ref) for ref in representative_trace.evidence_refs[:5]
                ],
            },
            "suspected_change": {
                "change_type": suspected_change.change_type,
                "change_ref": suspected_change.change_ref,
                "diff_ref": suspected_change.diff_ref,
                "summary": suspected_change.summary,
                "evidence_refs": [self._evidence_dict(ref) for ref in suspected_change.evidence_refs[:3]],
            },
            "timeline": timeline_snapshot,
        }

        def _planner(context: dict[str, Any]) -> dict[str, Any]:
            merged_context = dict(context)
            merged_context.update(planner_seed)
            return planner(merged_context)

        tool_registry = ToolRegistry(inspection_api=self._inspection_api or self._resolve_inspection_api(request))
        sandbox_guard = SandboxGuard(allowed_tools=tool_registry.allowed_tools)
        recursive_loop = RecursiveLoop(tool_registry=tool_registry, sandbox_guard=sandbox_guard)
        loop_result = recursive_loop.run(
            actions=[],
            planner=_planner,
            budget=self._recursive_budget,
            objective=f"incident per-trace drilldown trace={representative_trace.trace_id}",
            parent_call_id=f"incident:trace:{representative_trace.trace_id}",
            input_ref_hash=hash_excerpt(representative_trace.trace_id),
        )
        usage_total.add(
            StructuredGenerationUsage(
                tokens_in=int(loop_result.usage.tokens_in),
                tokens_out=int(loop_result.usage.tokens_out),
                cost_usd=float(loop_result.usage.cost_usd),
            )
        )
        self._merge_recursive_loop_result(runtime_tracker, loop_result)

        if loop_result.status == "failed":
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                raise ModelOutputInvalidError(
                    loop_result.error_message or "Recursive incident planner output was invalid."
                )
            raise RuntimeError(loop_result.error_message or "Recursive incident trace drilldown failed.")

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        summary = str(payload.get("incident_summary") or "").strip()
        hypotheses_payload = payload.get("hypotheses")
        actions_payload = payload.get("recommended_actions")
        gaps_payload = payload.get("gaps")
        confidence = self._coerce_confidence(payload.get("confidence"), default=0.6)
        hypotheses = hypotheses_payload if isinstance(hypotheses_payload, list) else []
        actions = actions_payload if isinstance(actions_payload, list) else []
        llm_gaps = [str(item).strip() for item in gaps_payload or [] if str(item).strip()]
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or f"Per-trace recursive budget reached for trace {representative_trace.trace_id}."
            )
        return (
            summary,
            [item for item in hypotheses if isinstance(item, dict)],
            [item for item in actions if isinstance(item, dict)],
            llm_gaps,
            confidence,
        )

    def _recursive_cross_trace_synthesis(
        self,
        *,
        request: IncidentDossierRequest,
        representative_profiles: list[dict[str, Any]],
        raw_hypotheses: list[dict[str, Any]],
        raw_actions: list[dict[str, Any]],
        runtime_tracker: dict[str, Any],
        usage_total: StructuredGenerationUsage,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str], float]:
        model_client = self._resolve_model_client()
        model_provider = str(getattr(model_client, "model_provider", self.model_provider) or "openai")
        planner = StructuredActionPlanner(
            client=model_client,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt_id=_RECURSIVE_INCIDENT_PROMPT_ID,
        )
        self.prompt_template_hash = planner.prompt_template_hash
        runtime_tracker["recursive_used"] = True
        runtime_tracker["llm_used"] = True
        runtime_tracker["model_provider"] = model_provider

        representative_snapshot = [
            {
                "trace_id": str(profile.get("trace_id") or ""),
                "bucket": str(profile.get("bucket") or ""),
                "error_spans": int(profile.get("error_spans") or 0),
                "latency_ms": float(profile.get("latency_ms") or 0.0),
                "signature": profile.get("signature"),
            }
            for profile in representative_profiles
        ]
        normalized_hypotheses: list[dict[str, Any]] = []
        for item in raw_hypotheses[:20]:
            if not isinstance(item, dict):
                continue
            evidence_refs: list[dict[str, Any]] = []
            for evidence in item.get("evidence_refs") or []:
                if isinstance(evidence, EvidenceRef):
                    evidence_refs.append(self._evidence_dict(evidence))
                elif isinstance(evidence, dict):
                    evidence_refs.append({str(key): evidence[key] for key in evidence})
            normalized_hypotheses.append(
                {
                    "statement": str(item.get("statement") or ""),
                    "confidence": self._coerce_confidence(item.get("confidence"), default=0.6),
                    "evidence_refs": evidence_refs,
                }
            )
        normalized_actions: list[dict[str, Any]] = []
        for item in raw_actions[:20]:
            if not isinstance(item, dict):
                continue
            normalized_actions.append(
                {
                    "priority": self._coerce_action_priority(item.get("priority")),
                    "type": self._coerce_action_type(item.get("type")),
                    "action": str(item.get("action") or ""),
                }
            )
        planner_seed = {
            "engine_type": self.engine_type,
            "incident_scope": "cross_trace_synthesis",
            "project_name": request.project_name,
            "incident_window": {
                "start": request.time_window_start,
                "end": request.time_window_end,
            },
            "representative_profiles": representative_snapshot,
            "trace_hypotheses": normalized_hypotheses,
            "trace_actions": normalized_actions,
        }

        def _planner(context: dict[str, Any]) -> dict[str, Any]:
            merged_context = dict(context)
            merged_context.update(planner_seed)
            return planner(merged_context)

        tool_registry = ToolRegistry(inspection_api=self._inspection_api or self._resolve_inspection_api(request))
        sandbox_guard = SandboxGuard(allowed_tools=tool_registry.allowed_tools)
        recursive_loop = RecursiveLoop(tool_registry=tool_registry, sandbox_guard=sandbox_guard)
        loop_result = recursive_loop.run(
            actions=[],
            planner=_planner,
            budget=self._recursive_budget,
            objective="incident cross-trace synthesis",
            parent_call_id="incident:cross-trace",
            input_ref_hash=hash_excerpt(f"{request.project_name}:{request.time_window_start}:{request.time_window_end}"),
        )
        usage_total.add(
            StructuredGenerationUsage(
                tokens_in=int(loop_result.usage.tokens_in),
                tokens_out=int(loop_result.usage.tokens_out),
                cost_usd=float(loop_result.usage.cost_usd),
            )
        )
        self._merge_recursive_loop_result(runtime_tracker, loop_result)

        if loop_result.status == "failed":
            if loop_result.error_code == "MODEL_OUTPUT_INVALID":
                raise ModelOutputInvalidError(
                    loop_result.error_message or "Recursive incident cross-trace synthesis was invalid."
                )
            raise RuntimeError(loop_result.error_message or "Recursive incident cross-trace synthesis failed.")

        payload = loop_result.output if isinstance(loop_result.output, dict) else {}
        summary = str(payload.get("incident_summary") or "").strip()
        hypotheses_payload = payload.get("hypotheses")
        actions_payload = payload.get("recommended_actions")
        gaps_payload = payload.get("gaps")
        confidence = self._coerce_confidence(payload.get("confidence"), default=0.65)
        hypotheses = hypotheses_payload if isinstance(hypotheses_payload, list) else []
        actions = actions_payload if isinstance(actions_payload, list) else []
        llm_gaps = [str(item).strip() for item in gaps_payload or [] if str(item).strip()]
        if loop_result.status == "terminated_budget":
            llm_gaps.append(
                loop_result.budget_reason
                or "Cross-trace recursive synthesis reached a budget limit."
            )
        return (
            summary,
            [item for item in hypotheses if isinstance(item, dict)],
            [item for item in actions if isinstance(item, dict)],
            llm_gaps,
            confidence,
        )

    def run(self, request: IncidentDossierRequest) -> IncidentDossier:
        self._reset_runtime_signals()
        usage_total = StructuredGenerationUsage()
        runtime_tracker: dict[str, Any] = {
            "attempts": 0,
            "llm_calls": 0,
            "llm_used": False,
            "fallback_used": False,
            "recursive_used": False,
            "model_provider": self.model_provider,
            "tool_calls": 0,
            "depth_reached": 0,
            "runtime_state": "completed",
            "budget_reason": "",
            "state_trajectory": [],
            "subcall_metadata": [],
            "sandbox_violations": [],
        }

        inspection_api = self._resolve_inspection_api(request)
        gaps: list[str] = []

        try:
            trace_summaries: list[dict[str, Any]] = []
            if request.trace_ids_override:
                for trace_id in request.trace_ids_override:
                    try:
                        spans = inspection_api.list_spans(trace_id) or []
                    except Exception as exc:
                        spans = []
                        gaps.append(f"Failed to list spans for override trace {trace_id}: {exc}")
                    latency_ms = max(
                        [self._safe_float(span.get("latency_ms")) for span in spans] or [0.0]
                    )
                    start_time = (
                        sorted(
                            [str(span.get("start_time") or "") for span in spans if span.get("start_time")],
                        )[0]
                        if spans
                        else request.time_window_start
                    )
                    trace_summaries.append(
                        {
                            "trace_id": trace_id,
                            "start_time": start_time,
                            "end_time": request.time_window_end,
                            "latency_ms": latency_ms,
                        }
                    )
            else:
                try:
                    trace_summaries = inspection_api.list_traces(
                        request.project_name,
                        start_time=request.time_window_start,
                        end_time=request.time_window_end,
                        filter_expr=request.filter_expr,
                    ) or []
                except Exception as exc:
                    gaps.append(f"Failed to list traces for incident window: {exc}")
                    trace_summaries = []

            candidates: list[dict[str, Any]] = []
            for trace_summary in trace_summaries:
                trace_id = str(trace_summary.get("trace_id") or "")
                if not trace_id:
                    continue
                try:
                    spans = inspection_api.list_spans(trace_id) or []
                except Exception as exc:
                    spans = []
                    gaps.append(f"Failed to list spans for trace {trace_id}: {exc}")
                candidates.append(self._candidate_from_trace(trace_summary, spans))

            representative_profiles = self._choose_representative_profiles(
                candidates,
                override_trace_ids=request.trace_ids_override,
            )
            if not representative_profiles and candidates:
                fallback = dict(self._sort_latency_candidates(candidates)[0])
                fallback["bucket"] = "latency"
                fallback["bucket_score"] = float(fallback.get("latency_ms") or 0.0)
                representative_profiles = [fallback]
                gaps.append("Selection quotas produced no representatives; fallback to top latency trace.")
            if not representative_profiles:
                representative_profiles = [
                    {
                        "trace_id": request.trace_ids_override[0] if request.trace_ids_override else "trace-stub",
                        "bucket": "latency",
                        "bucket_score": 0.0,
                        "error_spans": 0,
                        "latency_ms": 0.0,
                        "root_span_id": "root-span",
                        "start_time": request.time_window_start,
                        "signature": ("unknown_service", "none", "none"),
                    }
                ]
                gaps.append("No candidate traces were available; incident dossier used fallback trace.")

            representative_traces: list[RepresentativeTrace] = []
            impacted_components: list[str] = []
            for profile in representative_profiles:
                evidence = self._evidence_from_profile(profile)
                service_name = str((profile.get("signature") or ("unknown_service", "", ""))[0])
                if service_name not in impacted_components:
                    impacted_components.append(service_name)
                representative_traces.append(
                    RepresentativeTrace(
                        trace_id=str(profile.get("trace_id") or "trace-stub"),
                        why_selected=(
                            f"{profile.get('bucket')}_bucket(score={float(profile.get('bucket_score') or 0.0):.2f}, "
                            f"error_spans={int(profile.get('error_spans') or 0)}, "
                            f"latency_ms={float(profile.get('latency_ms') or 0.0):.2f})"
                        ),
                        evidence_refs=[evidence],
                    )
                )

            lead_evidence = representative_traces[0].evidence_refs[0]
            timeline = [
                IncidentTimelineEvent(
                    timestamp=request.time_window_start,
                    event=(
                        f"Incident window opened for project {request.project_name}; "
                        f"selected {len(representative_traces)} representative traces."
                    ),
                    evidence_refs=[lead_evidence],
                )
            ]
            suspected_change, config_timeline_event, config_gaps = self._build_suspected_change(
                inspection_api=inspection_api,
                request=request,
                lead_evidence=lead_evidence,
            )
            gaps.extend(config_gaps)
            if config_timeline_event is not None:
                timeline.append(config_timeline_event)
            timeline.sort(key=lambda event: event.timestamp)

            non_ok_count = sum(
                1 for profile in representative_profiles if int(profile.get("error_spans") or 0) > 0
            )
            if non_ok_count > 0:
                hypothesis_statement = (
                    f"Error-heavy representative traces suggest concentrated failure patterns across "
                    f"{non_ok_count} selected traces."
                )
                hypothesis_confidence = 0.68
            else:
                hypothesis_statement = (
                    "High-latency representative traces dominate the incident window with limited explicit error spans."
                )
                hypothesis_confidence = 0.57
            deterministic_hypothesis = IncidentHypothesis(
                rank=1,
                statement=hypothesis_statement,
                evidence_refs=[lead_evidence],
                confidence=hypothesis_confidence,
            )
            deterministic_action = RecommendedAction(
                priority="P1",
                type="follow_up_fix",
                action=(
                    "Inspect representative traces in bucket order and validate the latest config diff "
                    "against failing spans."
                ),
            )
            summary = (
                f"Selected {len(representative_traces)} representative traces "
                f"using deterministic error/latency/cluster buckets"
                f"{'; config diff correlated' if suspected_change.diff_ref else '; config diff unavailable'}."
            )
            confidence = min(0.8, 0.45 + (0.05 * len(representative_traces)))

            hypotheses: list[IncidentHypothesis] = [deterministic_hypothesis]
            actions: list[RecommendedAction] = [deterministic_action]

            if self._use_llm_judgment:
                raw_hypotheses: list[dict[str, Any]] = []
                raw_actions: list[dict[str, Any]] = []
                llm_summaries: list[str] = []
                llm_confidences: list[float] = []

                for profile, representative_trace in zip(representative_profiles, representative_traces):
                    try:
                        if self._use_recursive_runtime:
                            (
                                trace_summary,
                                trace_hypotheses,
                                trace_actions,
                                trace_gaps,
                                trace_confidence,
                            ) = self._recursive_trace_synthesis(
                                request=request,
                                profile=profile,
                                representative_trace=representative_trace,
                                timeline=timeline,
                                suspected_change=suspected_change,
                                runtime_tracker=runtime_tracker,
                                usage_total=usage_total,
                            )
                        else:
                            runtime_tracker["llm_calls"] = int(runtime_tracker.get("llm_calls") or 0) + 1
                            (
                                trace_summary,
                                trace_hypotheses,
                                trace_actions,
                                trace_gaps,
                                trace_confidence,
                                attempt_count,
                                usage,
                                model_provider,
                            ) = self._llm_trace_synthesis(
                                request=request,
                                profile=profile,
                                representative_trace=representative_trace,
                                timeline=timeline,
                                suspected_change=suspected_change,
                            )
                            runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + max(
                                1, int(attempt_count)
                            )
                            usage_total.add(usage)
                            runtime_tracker["llm_used"] = True
                            runtime_tracker["model_provider"] = model_provider
                        if trace_summary:
                            llm_summaries.append(trace_summary)
                        llm_confidences.append(trace_confidence)
                        if trace_gaps:
                            gaps.extend(trace_gaps)

                        trace_evidence = self._dedupe_evidence_refs(
                            list(representative_trace.evidence_refs) + list(suspected_change.evidence_refs)
                        )
                        for item in trace_hypotheses:
                            statement = str(item.get("statement") or "").strip()
                            if not statement:
                                continue
                            raw_hypotheses.append(
                                {
                                    "statement": statement,
                                    "confidence": self._coerce_confidence(
                                        item.get("confidence"),
                                        default=trace_confidence,
                                    ),
                                    "evidence_refs": trace_evidence,
                                }
                            )
                        for item in trace_actions:
                            action_text = str(item.get("action") or "").strip()
                            if not action_text:
                                continue
                            raw_actions.append(
                                {
                                    "priority": self._coerce_action_priority(item.get("priority")),
                                    "type": self._coerce_action_type(item.get("type")),
                                    "action": action_text,
                                }
                            )
                    except ModelOutputInvalidError as exc:
                        if self._use_recursive_runtime:
                            runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + 1
                        else:
                            runtime_tracker["attempts"] = int(runtime_tracker.get("attempts") or 0) + int(
                                getattr(exc, "attempt_count", 1) or 1
                            )
                            usage = getattr(exc, "usage", None)
                            if isinstance(usage, StructuredGenerationUsage):
                                usage_total.add(usage)
                            runtime_tracker["model_provider"] = str(
                                getattr(self._model_client, "model_provider", self.model_provider)
                                or self.model_provider
                            )
                        if not self._fallback_on_llm_error:
                            raise
                        runtime_tracker["fallback_used"] = True
                        gaps.append(
                            "LLM incident synthesis failed for representative trace "
                            f"{representative_trace.trace_id}; deterministic fallback was used: {exc}"
                        )

                if self._use_recursive_runtime:
                    try:
                        (
                            cross_summary,
                            cross_hypotheses,
                            cross_actions,
                            cross_gaps,
                            cross_confidence,
                        ) = self._recursive_cross_trace_synthesis(
                            request=request,
                            representative_profiles=representative_profiles,
                            raw_hypotheses=raw_hypotheses,
                            raw_actions=raw_actions,
                            runtime_tracker=runtime_tracker,
                            usage_total=usage_total,
                        )
                        if cross_summary:
                            llm_summaries.append(cross_summary)
                        llm_confidences.append(cross_confidence)
                        if cross_gaps:
                            gaps.extend(cross_gaps)
                        cross_evidence = self._dedupe_evidence_refs(
                            list(representative_traces[0].evidence_refs)
                            + list(suspected_change.evidence_refs)
                        )
                        for item in cross_hypotheses:
                            statement = str(item.get("statement") or "").strip()
                            if not statement:
                                continue
                            raw_hypotheses.append(
                                {
                                    "statement": statement,
                                    "confidence": self._coerce_confidence(
                                        item.get("confidence"),
                                        default=cross_confidence,
                                    ),
                                    "evidence_refs": cross_evidence,
                                }
                            )
                        for item in cross_actions:
                            action_text = str(item.get("action") or "").strip()
                            if not action_text:
                                continue
                            raw_actions.append(
                                {
                                    "priority": self._coerce_action_priority(item.get("priority")),
                                    "type": self._coerce_action_type(item.get("type")),
                                    "action": action_text,
                                }
                            )
                    except ModelOutputInvalidError as exc:
                        if not self._fallback_on_llm_error:
                            raise
                        runtime_tracker["fallback_used"] = True
                        gaps.append(
                            "LLM incident cross-trace synthesis failed; deterministic fallback was used: "
                            f"{exc}"
                        )

                if raw_hypotheses:
                    merged_hypotheses: dict[str, dict[str, Any]] = {}
                    for item in raw_hypotheses:
                        statement = str(item["statement"])
                        existing = merged_hypotheses.get(statement)
                        evidence_refs = item["evidence_refs"]
                        if existing is None:
                            merged_hypotheses[statement] = {
                                "statement": statement,
                                "confidence": float(item["confidence"]),
                                "evidence_refs": list(evidence_refs),
                            }
                            continue
                        existing["confidence"] = max(float(existing["confidence"]), float(item["confidence"]))
                        existing["evidence_refs"] = self._dedupe_evidence_refs(
                            list(existing["evidence_refs"]) + list(evidence_refs)
                        )
                    ordered_hypotheses = sorted(
                        merged_hypotheses.values(),
                        key=lambda row: (-float(row["confidence"]), str(row["statement"])),
                    )
                    hypotheses = [
                        IncidentHypothesis(
                            rank=index,
                            statement=str(row["statement"]),
                            evidence_refs=self._dedupe_evidence_refs(
                                [ref for ref in row["evidence_refs"] if isinstance(ref, EvidenceRef)]
                            )
                            or [lead_evidence],
                            confidence=self._coerce_confidence(row["confidence"], default=0.6),
                        )
                        for index, row in enumerate(ordered_hypotheses, start=1)
                    ]

                if raw_actions:
                    seen_actions: set[tuple[str, str, str]] = set()
                    normalized_actions: list[RecommendedAction] = []
                    for item in raw_actions:
                        key = (
                            str(item["priority"]),
                            str(item["type"]),
                            str(item["action"]),
                        )
                        if key in seen_actions:
                            continue
                        seen_actions.add(key)
                        normalized_actions.append(
                            RecommendedAction(
                                priority=str(item["priority"]),
                                type=str(item["type"]),
                                action=str(item["action"]),
                            )
                        )
                    actions = normalized_actions or [deterministic_action]

                if llm_summaries:
                    cleaned_summaries = [item for item in llm_summaries if item]
                    if self._use_recursive_runtime and cleaned_summaries:
                        selected_summaries = [cleaned_summaries[-1]]
                        if len(cleaned_summaries) > 1:
                            selected_summaries.insert(0, cleaned_summaries[0])
                    else:
                        selected_summaries = cleaned_summaries[:2]
                    if selected_summaries:
                        summary = f"{summary} LLM synthesis: {' | '.join(selected_summaries)}"
                if llm_confidences:
                    confidence = self._coerce_confidence(
                        sum(llm_confidences) / max(1, len(llm_confidences)),
                        default=confidence,
                    )

            return IncidentDossier(
                incident_summary=summary,
                impacted_components=impacted_components or ["unknown_service"],
                timeline=timeline,
                representative_traces=representative_traces,
                suspected_change=suspected_change,
                hypotheses=hypotheses,
                recommended_actions=actions,
                confidence=confidence,
                gaps=gaps,
            )
        finally:
            self._update_runtime_signals(
                usage_total=usage_total,
                runtime_tracker=runtime_tracker,
            )
