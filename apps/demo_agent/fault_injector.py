# ABOUTME: Runs seeded fault-injection trace generation and resolves resulting trace IDs from Phoenix.
# ABOUTME: Prefers live LlamaIndex fault runs and falls back to deterministic seeded traces when live mode is unavailable.

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv
from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from phoenix.session.client import Client

from apps.demo_agent.phase1_seeded_failures import emit_seeded_traces, export_spans_to_parquet


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROJECT_NAME = "phase1-seeded-failures"
DEFAULT_LOOKUP_LIMIT = 100000
DEFAULT_EXPORT_PATH = Path("datasets/seeded_failures/exports/spans.parquet")
DEFAULT_LIVE_MODEL = "gpt-4o-mini"
DEFAULT_LIVE_EMBED_MODEL = "text-embedding-3-small"


FAULT_PROFILE_TO_LABEL = {
    "profile_tool_failure": "tool_failure",
    "profile_retrieval_failure": "retrieval_failure",
    "profile_instruction_failure": "instruction_failure",
    "profile_upstream_dependency_failure": "upstream_dependency_failure",
    "profile_data_schema_mismatch": "data_schema_mismatch",
}


class LiveFaultInjectionUnavailableError(RuntimeError):
    """Raised when live LlamaIndex fault execution cannot be started in this environment."""


class _LiveDeps:
    def __init__(
        self,
        *,
        document_cls: Any,
        vector_index_cls: Any,
        settings_cls: Any,
        openai_llm_cls: Any,
        openai_embed_cls: Any,
        instrumentor_cls: Any,
    ) -> None:
        self.Document = document_cls
        self.VectorStoreIndex = vector_index_cls
        self.Settings = settings_cls
        self.OpenAI = openai_llm_cls
        self.OpenAIEmbedding = openai_embed_cls
        self.LlamaIndexInstrumentor = instrumentor_cls


def _base_phoenix_endpoint(endpoint: str) -> str:
    value = str(endpoint or "http://127.0.0.1:6006").strip()
    if not value:
        return "http://127.0.0.1:6006"
    if value.endswith("/v1/traces"):
        value = value[: -len("/v1/traces")]
    return value.rstrip("/")


def _collector_traces_endpoint(endpoint: str) -> str:
    base = _base_phoenix_endpoint(endpoint)
    return f"{base}/v1/traces"


def _single_case_manifest(*, fault_profile: str, run_id: str) -> dict[str, Any]:
    if fault_profile not in FAULT_PROFILE_TO_LABEL:
        supported = ", ".join(sorted(FAULT_PROFILE_TO_LABEL))
        raise ValueError(f"Unsupported fault profile '{fault_profile}'. Supported: {supported}")
    return {
        "dataset_id": "seeded_failures_v1",
        "generator_version": "0.1.0",
        "seed": None,
        "cases": [
            {
                "run_id": run_id,
                "trace_id": None,
                "expected_label": FAULT_PROFILE_TO_LABEL[fault_profile],
                "fault_profile": fault_profile,
                "notes": f"seeded deterministic case for {run_id}",
            }
        ],
    }


def _resolve_run_id_column(columns: list[str]) -> str | None:
    preferred = [
        "attributes.phase1.run_id",
        "attributes.phase1",
        "phase1.run_id",
        "run_id",
    ]
    for name in preferred:
        if name in columns:
            return name
    for name in columns:
        if "phase1.run_id" in name:
            return name
    for name in columns:
        if name.endswith("run_id"):
            return name
    return None


def _resolve_trace_id_column(columns: list[str]) -> str | None:
    if "context.trace_id" in columns:
        return "context.trace_id"
    if "trace_id" in columns:
        return "trace_id"
    for name in columns:
        if name.endswith("trace_id"):
            return name
    return None


def _resolve_sort_column(columns: list[str]) -> str | None:
    for candidate in ("start_time", "start_time_unix_nano", "start_time_utc"):
        if candidate in columns:
            return candidate
    return None


def _normalize_run_id_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        run_id_value = value.get("run_id")
        if run_id_value is None:
            return None
        normalized = str(run_id_value).strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _lookup_trace_ids_by_run_id(
    *,
    endpoint: str,
    project_name: str,
    run_ids: list[str],
    limit: int = DEFAULT_LOOKUP_LIMIT,
) -> dict[str, str]:
    if not run_ids:
        return {}

    client = Client(endpoint=_base_phoenix_endpoint(endpoint))
    dataframe = client.get_spans_dataframe(project_name=project_name, limit=limit)
    if dataframe is None or dataframe.empty:
        return {}

    run_id_column = _resolve_run_id_column(list(dataframe.columns))
    trace_id_column = _resolve_trace_id_column(list(dataframe.columns))
    if run_id_column is None or trace_id_column is None:
        return {}

    subset = dataframe[[run_id_column, trace_id_column]].copy()
    sort_column = _resolve_sort_column(list(dataframe.columns))
    if sort_column is not None:
        subset[sort_column] = dataframe[sort_column]
        subset = subset.sort_values(by=sort_column, kind="stable")
    subset = subset.dropna(subset=[run_id_column, trace_id_column])

    target_ids = {str(value) for value in run_ids}
    mapping: dict[str, str] = {}
    for _, row in subset.iterrows():
        run_id_value = _normalize_run_id_value(row[run_id_column])
        if run_id_value is None or run_id_value not in target_ids:
            continue
        mapping[run_id_value] = str(row[trace_id_column])
    return mapping


def _resolve_live_dependencies() -> _LiveDeps:
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    except Exception as exc:  # noqa: BLE001
        raise LiveFaultInjectionUnavailableError(
            "Live LlamaIndex fault path requires llama-index and openinference instrumentation packages."
        ) from exc

    return _LiveDeps(
        document_cls=Document,
        vector_index_cls=VectorStoreIndex,
        settings_cls=Settings,
        openai_llm_cls=OpenAI,
        openai_embed_cls=OpenAIEmbedding,
        instrumentor_cls=LlamaIndexInstrumentor,
    )


def _live_model_name() -> str:
    value = os.getenv("PHASE1_LIVE_LLM_MODEL", DEFAULT_LIVE_MODEL).strip()
    return value or DEFAULT_LIVE_MODEL


def _live_embed_model_name() -> str:
    value = os.getenv("PHASE1_LIVE_EMBED_MODEL", DEFAULT_LIVE_EMBED_MODEL).strip()
    return value or DEFAULT_LIVE_EMBED_MODEL


def _load_local_env_file() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def _resolve_trace_id_with_retry(
    *,
    endpoint: str,
    project_name: str,
    run_id: str,
    limit: int = DEFAULT_LOOKUP_LIMIT,
    attempts: int = 8,
    sleep_sec: float = 0.5,
) -> str | None:
    max_attempts = max(1, int(attempts))
    for _ in range(max_attempts):
        mapping = _lookup_trace_ids_by_run_id(
            endpoint=endpoint,
            project_name=project_name,
            run_ids=[run_id],
            limit=limit,
        )
        trace_id = mapping.get(run_id)
        if trace_id:
            return trace_id
        time.sleep(max(0.0, float(sleep_sec)))
    return None


def _ensure_openai_api_key_for_live_path() -> None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise LiveFaultInjectionUnavailableError(
            "OPENAI_API_KEY is required for the live LlamaIndex fault path."
        )


def _profile_documents(fault_profile: str) -> list[str]:
    base_docs = [
        "Chinook Track table includes thousands of tracks and metadata fields.",
        "The SQL answer for the Track table count is 3503 in this benchmark dataset.",
        "Tooling often queries Track and Album tables to answer counting questions.",
    ]
    if fault_profile == "profile_retrieval_failure":
        return [
            "Warehouse forklift maintenance schedule for Q3.",
            "Beverage inventory levels by region and depot.",
            "Employee travel reimbursement policy appendix.",
        ]
    return base_docs


def _profile_query(fault_profile: str) -> str:
    if fault_profile == "profile_instruction_failure":
        return (
            "Return ONLY strict JSON with keys answer and confidence. "
            "Do not include any prose. What is the Track table count?"
        )
    if fault_profile == "profile_data_schema_mismatch":
        return "Return the Track table count as a plain sentence."
    return "How many tracks are in the Track table?"


def _inject_profile_fault_markers(*, fault_profile: str, tracer: Any, response_text: str) -> None:
    if fault_profile == "profile_tool_failure":
        with tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("phase1.step", "tool.call")
            span.set_status(Status(StatusCode.ERROR, "forced live tool timeout"))
        return

    if fault_profile == "profile_retrieval_failure":
        with tracer.start_as_current_span("retriever.fetch") as span:
            span.set_attribute("phase1.step", "retriever.fetch")
            span.set_attribute("phase1.retrieval.relevance", 0.08)
            span.set_status(Status(StatusCode.OK))
        return

    if fault_profile == "profile_instruction_failure":
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("phase1.step", "llm.generate")
            span.set_attribute("phase1.output.format", "unexpected")
            span.set_status(Status(StatusCode.ERROR, "format drift"))
        return

    if fault_profile == "profile_upstream_dependency_failure":
        with tracer.start_as_current_span("dependency.http") as span:
            span.set_attribute("phase1.step", "dependency.http")
            span.set_attribute("http.status_code", 503)
            span.set_status(Status(StatusCode.ERROR, "upstream unavailable"))
        return

    if fault_profile == "profile_data_schema_mismatch":
        with tracer.start_as_current_span("tool.parse") as span:
            span.set_attribute("phase1.step", "tool.parse")
            span.set_attribute("phase1.response_preview", response_text[:120])
            span.set_status(Status(StatusCode.ERROR, "schema mismatch"))


def _run_live_llamaindex_fault(
    *,
    fault_profile: str,
    run_id: str,
    phoenix_endpoint: str,
    project_name: str,
) -> str:
    deps = _resolve_live_dependencies()
    _ensure_openai_api_key_for_live_path()

    tracer_provider = register(
        endpoint=_collector_traces_endpoint(phoenix_endpoint),
        protocol="http/protobuf",
        project_name=project_name,
        batch=False,
        verbose=False,
    )

    instrumentor = deps.LlamaIndexInstrumentor()
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001
        raise LiveFaultInjectionUnavailableError(
            "Failed to instrument LlamaIndex for Phoenix tracing."
        ) from exc

    try:
        deps.Settings.llm = deps.OpenAI(model=_live_model_name(), temperature=0)
        deps.Settings.embed_model = deps.OpenAIEmbedding(model=_live_embed_model_name())

        docs = [deps.Document(text=text) for text in _profile_documents(fault_profile)]
        index = deps.VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(similarity_top_k=2)

        tracer = trace_api.get_tracer("phase10.live_fault_injector")
        with tracer.start_as_current_span("agent.run") as root:
            root.set_attribute("phase1.run_id", run_id)
            root.set_attribute("phase1.project", project_name)
            root.set_attribute("phase1.fault_profile", fault_profile)

            response = query_engine.query(_profile_query(fault_profile))
            response_text = str(response)
            root.set_attribute("phase1.response_length", len(response_text))
            _inject_profile_fault_markers(
                fault_profile=fault_profile,
                tracer=tracer,
                response_text=response_text,
            )

        tracer_provider.force_flush()
        time.sleep(1.0)
    finally:
        try:
            instrumentor.uninstrument()
        except Exception:  # noqa: BLE001
            pass

    trace_id = _resolve_trace_id_with_retry(
        endpoint=phoenix_endpoint,
        project_name=project_name,
        run_id=run_id,
        attempts=8,
        sleep_sec=0.5,
    )
    if not trace_id:
        raise RuntimeError(
            f"Live LlamaIndex run completed but trace_id could not be resolved for run_id '{run_id}'."
        )
    return trace_id


def _run_seeded_fallback_fault(
    *,
    fault_profile: str,
    run_id: str,
    phoenix_endpoint: str,
    project_name: str,
    lookup_limit: int,
) -> str:
    manifest = _single_case_manifest(fault_profile=fault_profile, run_id=run_id)
    emit_seeded_traces(
        manifest,
        project_name=project_name,
        endpoint=_collector_traces_endpoint(phoenix_endpoint),
    )
    trace_id = _resolve_trace_id_with_retry(
        endpoint=phoenix_endpoint,
        project_name=project_name,
        run_id=run_id,
        limit=lookup_limit,
        attempts=8,
        sleep_sec=0.5,
    )
    if not trace_id:
        raise RuntimeError(
            f"Trace ID could not be resolved for run_id '{run_id}' in project '{project_name}'."
        )
    return trace_id


def run_with_fault(
    *,
    fault_profile: str,
    run_id: str,
    phoenix_endpoint: str = "http://127.0.0.1:6006",
    project_name: str = DEFAULT_PROJECT_NAME,
    lookup_limit: int = DEFAULT_LOOKUP_LIMIT,
    live_only: bool = False,
) -> str:
    _load_local_env_file()
    _single_case_manifest(fault_profile=fault_profile, run_id=run_id)

    try:
        return _run_live_llamaindex_fault(
            fault_profile=fault_profile,
            run_id=run_id,
            phoenix_endpoint=phoenix_endpoint,
            project_name=project_name,
        )
    except LiveFaultInjectionUnavailableError:
        if live_only:
            raise
        return _run_seeded_fallback_fault(
            fault_profile=fault_profile,
            run_id=run_id,
            phoenix_endpoint=phoenix_endpoint,
            project_name=project_name,
            lookup_limit=lookup_limit,
        )
    except Exception as live_exc:  # noqa: BLE001
        if live_only:
            raise
        try:
            return _run_seeded_fallback_fault(
                fault_profile=fault_profile,
                run_id=run_id,
                phoenix_endpoint=phoenix_endpoint,
                project_name=project_name,
                lookup_limit=lookup_limit,
            )
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "Both live LlamaIndex and deterministic fallback paths failed. "
                f"live_error={live_exc}; fallback_error={fallback_exc}"
            ) from fallback_exc


def run_all_seeded_failures(
    *,
    manifest_path: str = "datasets/seeded_failures/manifest.json",
    phoenix_endpoint: str = "http://127.0.0.1:6006",
    project_name: str = DEFAULT_PROJECT_NAME,
    export_path: Path = DEFAULT_EXPORT_PATH,
    lookup_limit: int = DEFAULT_LOOKUP_LIMIT,
    live_only: bool = False,
) -> dict[str, str]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    run_to_trace: dict[str, str] = {}
    for case in manifest.get("cases", []):
        run_id = str(case.get("run_id", "")).strip()
        fault_profile = str(case.get("fault_profile", "")).strip()
        if not run_id:
            raise ValueError("Each manifest case must include a non-empty run_id.")
        trace_id = run_with_fault(
            fault_profile=fault_profile,
            run_id=run_id,
            phoenix_endpoint=phoenix_endpoint,
            project_name=project_name,
            lookup_limit=lookup_limit,
            live_only=live_only,
        )
        case["trace_id"] = trace_id
        run_to_trace[run_id] = trace_id

    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    export_spans_to_parquet(
        endpoint=_base_phoenix_endpoint(phoenix_endpoint),
        project_name=project_name,
        output_path=Path(export_path),
        limit=lookup_limit,
    )
    return run_to_trace
