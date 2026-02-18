You are my project copilot. We are building an RLM-powered investigation layer on top of Arize Phoenix.

Important constraints / changes
- Do not use reserved internal project names. Use a neutral project name like “Phoenix-RLM Investigator” and use it consistently.
- We have a blank canvas except: (1) the Recursive Language Models (RLM) paper and (2) the Arize Phoenix repo/docs. Our FIRST job is to clone an example agent (preferably from Phoenix tutorials) so we can generate traces, then run evals on those traces. Only after that do we build the investigator.

What we are building first (scope)
We will deliver TWO capabilities first:
(2) Trace RCA across distributed agent traces (Phoenix-native)
(3) Incident investigation with rich context (start with trace + deploy/config context; then add logs/metrics connectors)

Key references you must use when planning
- Phoenix tracing tutorial + “Your First Traces” (how to instrument and view traces). Use this as the setup guide. 
- Phoenix “Running evals on traces” (how to export trace datasets and write eval results back into Phoenix UI).
- Phoenix tutorial notebooks in the Phoenix repo for an example agent that already works with Phoenix tracing:
  - LangChain agent tracing tutorial notebook
  - LlamaIndex OpenAI agent tracing tutorial notebook
  - LangGraph agent tracing tutorial notebook
- OpenTelemetry signals separation (traces vs metrics vs logs): this is why incident “rich context” needs connectors.
- RLM definition + DSPy RLM (sandboxed REPL, recursive subcalls): our investigator should treat traces (and later logs/metrics) as an external environment.

Phase 1: Get traces working (must happen before anything else)
Goal: have a runnable agent that emits traces into a locally running Phoenix instance.

Tasks:
1) Stand up Phoenix locally using the official terminal instructions (no Docker). Confirm Phoenix UI is accessible and the collector endpoint is reachable.
2) Clone the Arize Phoenix repo and pick ONE tutorial notebook as the starting agent:
   - Prefer langgraph_agent_tracing_tutorial.ipynb if we want a more “systems” feel,
   - otherwise langchain_agent_tracing_tutorial.ipynb or llama_index_openai_agent_tracing_tutorial.ipynb for fastest setup.
3) Run the chosen tutorial end-to-end and confirm traces appear in Phoenix (LLM calls + tool calls + retrieval if applicable).
4) Generate a trace dataset with intentional failure cases (at least 30–100 traces):
   - tool errors (force a tool failure / timeout)
   - retrieval mistakes (wrong doc / irrelevant chunk)
   - ambiguous entity resolution (two similar entities)
   - prompt regression (change instruction and cause format drift)
Acceptance criteria for Phase 1:
- Traces are visible, navigable, and filterable in Phoenix, with consistent span structure.
- We have a saved dataset (or easily reproducible script) that can generate the same failure traces repeatedly.

Phase 2: Build Trace RCA (use case 2)
Goal: given a trace, produce a structured root-cause analysis with evidence pointers, and write it back into Phoenix as eval results/annotations.

Design requirements:
- RCA output must be structured JSON.
- RCA must cite evidence by stable IDs: trace_id, span_id, artifact_id (retrieved chunk IDs, tool call IDs).
- RCA should be reproducible: rerunning on the same trace dataset yields comparable results.

Implementation plan:
1) Define an RCA taxonomy (keep it simple but useful):
   - Retrieval failure
   - Tool failure
   - Model/prompt instruction failure
   - Upstream dependency failure (API errors/timeouts)
   - Data/schema mismatch (tool output parsing)
2) Build a “Trace Inspection API” (read-only) that the evaluator can call:
   - list_spans(trace_id)
   - get_span(span_id) (attrs, timings, status)
   - get_children(span_id)
   - get_tool_io(span_id)
   - get_retrieval_chunks(span_id)
   - search_trace(trace_id, pattern)
3) Build a baseline RCA evaluator using Phoenix “evals on traces”:
   - deterministic narrowing first (find hot spans: errors, retries, slowest spans)
   - then an LLM judge to produce structured RCA JSON
4) Log RCA results back into Phoenix using the evals-on-traces workflow so the RCA appears in the UI on each trace.

Acceptance criteria for Phase 2:
- For our seeded failure dataset, RCA labels match expectations on the majority of cases.
- RCA outputs show clear evidence pointers (span IDs) that a reviewer can click through in Phoenix.

Phase 3: Build Incident Investigation with “rich context” (use case 3)
Goal: given an incident trigger, produce an incident dossier that correlates traces with additional context.

Important: “rich context” is staged.
- V1 (still “rich”, no external backend required): correlate traces + deploy/config diffs + runbook notes (stored in-repo).
- V2 (full observability): add log + metric connectors (because OpenTelemetry treats logs/metrics as separate signals from traces).

Inputs (incident trigger):
- time window
- app/service name
- symptom (e.g., error spike, latency spike, hallucination spike, policy violation spike)
- set of trace IDs (or a query to fetch representative traces)

Dossier output schema (structured JSON):
- incident_summary
- impacted_components
- timeline (key events with timestamps)
- representative_traces (IDs + why selected)
- suspected_change (deploy/config diff reference)
- correlated_signals:
   - traces (span evidence)
   - logs (top signatures, sampled exemplars) [V2]
   - metrics (time series slices) [V2]
- hypotheses (ranked, each with evidence pointers)
- recommended_actions (immediate mitigation + follow-up fixes)
- confidence + gaps (what evidence was missing)

Implementation plan:
1) Build a “Trace cluster selector”:
   - pick N representative traces from the incident window (errors, p95 latency, or semantic similarity)
2) Add deploy/config context:
   - store a “last known good” config snapshot in repo
   - store a “current” snapshot
   - compute diffs (commit hash or file diff) and treat as evidence objects
3) Build the dossier generator as an evaluator:
   - coarse-to-fine: summarize the window → drill into top traces → pull specific span evidence → attach deploy/config diffs
4) (Optional but recommended) Add minimal connectors for V2:
   - Logs connector: a simple local structured log store (or OpenTelemetry logs) that can be queried by trace_id/time window.
   - Metrics connector: a simple time series store (or Prometheus) queried by metric + time window.
   These become tools callable by the investigator.

Acceptance criteria for Phase 3:
- Given an incident window, the system produces a coherent dossier with a timeline + trace evidence + suspected change and clear next actions.
- The dossier is written back into Phoenix as an artifact/annotation linked to the incident’s representative traces.

Phase 4: Upgrade the investigator to RLM (RLM-powered investigation engine)
Goal: replace “stuff everything into one prompt” with an RLM that treats traces (and later logs/metrics) as an external environment.

Rules:
- Use DSPy’s RLM module (sandboxed Python REPL + recursive subcalls).
- The REPL must be sandboxed: no network, no filesystem. It can only call our Inspection APIs (trace/log/metric/config).
- The RLM must output the same structured schemas as the baseline (RCA JSON, dossier JSON) so it’s drop-in.

How RLM is used:
- The RLM will programmatically explore a trace:
  - identify hot spans,
  - recursively inspect only suspicious tool outputs and retrieval chunks,
  - compare across multiple traces by partitioning and mapping,
  - synthesize a final RCA/dossier with evidence pointers.

Your first response back to me must contain
1) The chosen project name + a 1-page plan with phases and acceptance criteria.
2) The specific tutorial notebook you recommend cloning FIRST (from the Phoenix repo) and why.
3) The exact RCA taxonomy + RCA output JSON schema.
4) The incident dossier JSON schema.
5) The “Trace Inspection API” spec (function signatures + returned object structure).
6) A minimal list of span fields we must capture to support RCA and dossier (what we must ensure OpenInference/OTEL emits).
7) A risk list (top 5 technical risks) + mitigations.

Core sources (use these while planning)
- Phoenix terminal setup for local run.
- Phoenix tracing tutorial and “Your First Traces.”
- Phoenix tutorial notebooks (LangChain/LlamaIndex/LangGraph agent tracing).
- Phoenix “Running evals on traces.”
- OpenTelemetry signals doc (traces/metrics/logs separation).
- RLM paper + DSPy RLM docs.
