# ABOUTME: Implements deterministic incident representative trace selection over the Inspection API.
# ABOUTME: Produces contract-aligned incident dossiers with evidence-linked trace selection rationale.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from investigator.inspection_api import PhoenixInspectionAPI
from investigator.runtime.prompt_registry import get_prompt_definition
from investigator.runtime.contracts import (
    EvidenceRef,
    IncidentDossier,
    IncidentHypothesis,
    IncidentTimelineEvent,
    InputRef,
    RecommendedAction,
    RepresentativeTrace,
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


class IncidentDossierEngine:
    engine_type = "incident_dossier"
    output_contract_name = "IncidentDossier"
    engine_version = "0.2.0"
    model_provider = "openai"
    model_name = "gpt-5-mini"
    prompt_template_hash = _INCIDENT_PROMPT.prompt_template_hash
    temperature = 0.0

    def __init__(
        self,
        inspection_api: Any | None = None,
        *,
        max_representatives: int = 12,
        error_quota: int = 5,
        latency_quota: int = 5,
        cluster_quota: int = 2,
    ) -> None:
        self._inspection_api = inspection_api
        self._max_representatives = max_representatives
        self._error_quota = error_quota
        self._latency_quota = latency_quota
        self._cluster_quota = cluster_quota

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

    def run(self, request: IncidentDossierRequest) -> IncidentDossier:
        inspection_api = self._resolve_inspection_api(request)
        gaps: list[str] = []

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
        hypothesis = IncidentHypothesis(
            rank=1,
            statement=hypothesis_statement,
            evidence_refs=[lead_evidence],
            confidence=hypothesis_confidence,
        )
        action = RecommendedAction(
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
        return IncidentDossier(
            incident_summary=summary,
            impacted_components=impacted_components or ["unknown_service"],
            timeline=timeline,
            representative_traces=representative_traces,
            suspected_change=suspected_change,
            hypotheses=[hypothesis],
            recommended_actions=[action],
            confidence=confidence,
            gaps=gaps,
        )
