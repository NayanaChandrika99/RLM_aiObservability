# ABOUTME: Runs a local LangGraph tutorial-style SQL agent with Phoenix tracing for Phase 1 validation.
# ABOUTME: Produces a deterministic readiness/output summary and confirms trace ingestion for the selected project.

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from phoenix.session.client import Client

from apps.demo_agent.phase1_langgraph_runner import (
    default_collector_endpoint,
    otlp_http_traces_endpoint,
    require_openai_api_key,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_PATH = REPO_ROOT / "apps" / "demo_agent" / "Chinook.db"
DEFAULT_PROJECT_NAME = "phase1-langgraph-tutorial"
DEFAULT_MODEL_NAME = "gpt-5-mini"
DEFAULT_QUESTION = "How many tracks are in the Track table? Use the database tools."


def tutorial_project_name() -> str:
    return os.getenv("PHASE1_PROJECT_NAME", DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME


def resolved_model_name() -> str:
    return os.getenv("PHASE1_LLM_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def _ensure_env_loaded() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def ensure_chinook_db() -> Path:
    if CHINOOK_PATH.exists():
        return CHINOOK_PATH
    CHINOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(CHINOOK_URL, timeout=30)
    response.raise_for_status()
    CHINOOK_PATH.write_bytes(response.content)
    return CHINOOK_PATH


def _register_langchain_tracing(project_name: str):
    tracer_provider = register(
        endpoint=otlp_http_traces_endpoint(),
        protocol="http/protobuf",
        project_name=project_name,
        batch=False,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


def _build_agent(model_name: str):
    db = SQLDatabase.from_uri(f"sqlite:///{ensure_chinook_db()}")
    llm = ChatOpenAI(model=model_name, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return create_react_agent(llm, tools)


def _extract_final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def run_langgraph_tutorial_trace(question: str | None = None) -> dict[str, Any]:
    _ensure_env_loaded()
    require_openai_api_key()
    project_name = tutorial_project_name()
    model_name = resolved_model_name()
    tracer_provider = _register_langchain_tracing(project_name)
    agent = _build_agent(model_name=model_name)
    user_question = question or DEFAULT_QUESTION
    result = agent.invoke({"messages": [("user", user_question)]})
    tracer_provider.force_flush()
    time.sleep(1.0)

    client = Client(endpoint=default_collector_endpoint())
    dataframe = client.get_spans_dataframe(project_name=project_name, limit=100000)
    span_count = 0 if dataframe is None else int(len(dataframe))
    trace_count = 0
    span_kinds: list[str] = []
    tool_span_count = 0
    if dataframe is not None and not dataframe.empty and "context.trace_id" in dataframe.columns:
        trace_count = int(dataframe["context.trace_id"].nunique())
    if dataframe is not None and not dataframe.empty and "span_kind" in dataframe.columns:
        span_kinds = sorted(dataframe["span_kind"].dropna().unique().tolist())
        tool_span_count = int((dataframe["span_kind"] == "TOOL").sum())

    return {
        "project_name": project_name,
        "model_name": model_name,
        "question": user_question,
        "collector_endpoint": default_collector_endpoint(),
        "otlp_endpoint": otlp_http_traces_endpoint(),
        "final_answer": _extract_final_text(result),
        "span_count": span_count,
        "trace_count": trace_count,
        "span_kinds": span_kinds,
        "tool_span_count": tool_span_count,
    }


def main() -> None:
    summary = run_langgraph_tutorial_trace()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
