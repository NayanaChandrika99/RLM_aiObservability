# ABOUTME: Defines Phase 1 helpers for selecting the LangGraph Phoenix tutorial and local collector settings.
# ABOUTME: Provides a readiness report and preflight checks before running the tutorial agent end-to-end.

from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_NOTEBOOK_RELATIVE_PATH = Path(
    "phoenix/tutorials/tracing/langgraph_agent_tracing_tutorial.ipynb"
)
DEFAULT_COLLECTOR_ENDPOINT = "http://127.0.0.1:6006"


def tutorial_notebook_path() -> Path:
    return REPO_ROOT / TUTORIAL_NOTEBOOK_RELATIVE_PATH


def default_collector_endpoint() -> str:
    value = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_COLLECTOR_ENDPOINT).strip()
    return value.rstrip("/") if value else DEFAULT_COLLECTOR_ENDPOINT


def otlp_http_traces_endpoint() -> str:
    endpoint = default_collector_endpoint()
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return f"{endpoint}/v1/traces"


def require_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "OPENAI_API_KEY is required for the LangGraph tutorial run. "
        "Set OPENAI_API_KEY and rerun."
    )


def readiness_report() -> dict[str, object]:
    notebook = tutorial_notebook_path()
    openai_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "tutorial_notebook": str(notebook),
        "tutorial_notebook_exists": notebook.exists(),
        "collector_endpoint": default_collector_endpoint(),
        "otlp_http_traces_endpoint": otlp_http_traces_endpoint(),
        "openai_key_present": openai_key_present,
        "recommended_model": os.getenv("PHASE1_LLM_MODEL", "gpt-5-mini"),
    }


def main() -> None:
    report = readiness_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["tutorial_notebook_exists"] or not report["openai_key_present"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
