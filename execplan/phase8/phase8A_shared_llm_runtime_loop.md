# Phase 8A: Shared LLM Runtime Loop (Real Model Calls + Structured Output)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

This phase introduces the first real LLM execution path used by all three engines. After completion, Nainy can run an evaluator and verify that runtime usage reflects actual model activity (`tokens_in`, `tokens_out`, and cost) rather than metadata placeholders. The engines still keep deterministic evidence selection in place, but now have one shared runtime function that can request schema-constrained model output and return contract-safe data.

"Shared runtime loop" in this phase means a reusable model invocation loop with:

- provider-agnostic client interface (OpenAI first implementation),
- deterministic prompt-template loading,
- JSON-schema constrained output,
- deterministic retry/error mapping,
- token/cost accounting in run artifacts.

Recursive delegation is explicitly deferred to Phase 8B.

## Progress

- [x] (2026-02-10 15:00Z) Reviewed runtime, contract, and engine references for model-call insertion points.
- [x] (2026-02-10 15:00Z) Drafted initial Phase 8A plan.
- [x] (2026-02-10 15:12Z) Revised Phase 8A with provider abstraction, structured output strategy, prompt versioning, cost caps, and test-isolation strategy.
- [x] (2026-02-10 15:35Z) Added shared runtime model client modules (`llm_client.py`, `llm_loop.py`) and prompt registry.
- [x] (2026-02-10 15:35Z) Added structured response validation/retry loop and mapped exhausted retries to `MODEL_OUTPUT_INVALID`.
- [x] (2026-02-10 15:35Z) Integrated runtime usage and cost accounting into `run_engine` artifacts (`cost_usd`, provider propagation, cost cap checks).
- [x] (2026-02-10 15:35Z) Added Phase 8 RED->GREEN tests for prompt hashing, LLM client/loop, runtime accounting, and RCA engine LLM path.
- [x] (2026-02-10 15:40Z) Ran a live RCA smoke call with `use_llm_judgment=True`; `run_record.json` persisted non-zero `tokens_in`, `tokens_out`, and `cost_usd`.
- [x] (2026-02-10 15:42Z) Fixed OpenAI `gpt-5-mini` temperature compatibility in shared client and re-ran live smoke call successfully with default engine settings.

## Surprises & Discoveries

- Observation: Current `RuntimeRef.model_name` is descriptive metadata only; run logic does not consume it.
  Evidence: `investigator/runtime/runner.py` reads engine attributes into `RuntimeRef` but never invokes a model client.
- Observation: Current contract locks provider to OpenAI.
  Evidence: `RuntimeRef.model_provider: Literal["openai"]` in `investigator/runtime/contracts.py`.
- Observation: Current prompt hash values are static string constants and are not tied to on-disk prompt content.
  Evidence: class attributes in `investigator/rca/engine.py`, `investigator/compliance/engine.py`, `investigator/incident/engine.py`.
- Observation: Failed structured-generation retries can lose token/cost accounting unless usage is attached to the raised runtime error path.
  Evidence: `tests/unit/test_trace_rca_engine_phase8_llm.py` initially failed with `tokens_in == 0` on `MODEL_OUTPUT_INVALID` until usage propagation was added.
- Observation: `gpt-5-mini` rejects `temperature` on Responses API, so client must omit the parameter for this model family.
  Evidence: Live smoke run at 2026-02-10 15:38Z failed with `400 Unsupported parameter: 'temperature'`; fixed via model-aware omission and validated by live rerun at 15:41Z.

## Decision Log

- Decision: Introduce one dedicated runtime model client module under `investigator/runtime/` and avoid direct SDK imports in engine files.
  Rationale: Keeps engine code focused on evidence selection and domain reasoning.
  Date/Author: 2026-02-10 / Codex
- Decision: Keep OpenAI as first concrete provider but define provider-agnostic runtime interfaces and metadata in this phase.
  Rationale: Enables future provider addition without redesigning run-record contracts.
  Date/Author: 2026-02-10 / Codex
- Decision: Use schema-constrained JSON output mode as the default strategy, not free-form text parsing.
  Rationale: Improves reliability, reduces repair retries, and simplifies deterministic validation.
  Date/Author: 2026-02-10 / Codex
- Decision: Persist prompt templates as versioned files and compute prompt hash from canonical prompt+schema bytes.
  Rationale: Reproducibility requires immutable template identity.
  Date/Author: 2026-02-10 / Codex
- Decision: Treat parse failures as first-class runtime failures (`MODEL_OUTPUT_INVALID`) and persist failed run records.
  Rationale: Contract requires run artifacts for all outcomes and explicit error taxonomy.
  Date/Author: 2026-02-10 / Codex

## Outcomes & Retrospective

Phase implemented for the shared runtime loop and one engine path (`TraceRCAEngine` opt-in LLM mode). Shared model invocation, structured retry behavior, prompt hashing, and runtime usage/cost persistence are now covered by unit tests and one live smoke run.

## References

The following files were reviewed fully before drafting and revising this plan; listed behaviors will be reused:

- `specs/rlm_runtime_contract.md`
  Behavior reused: runtime error taxonomy and budget expectations.
- `specs/formal_contracts.md`
  Behavior reused: RunRecord shape and compatibility rules.
- `API.md`
  Behavior reused: run artifact persistence requirements.
- `investigator/runtime/contracts.py`
  Behavior reused: dataclass schema for budget, usage, and run record serialization.
- `investigator/runtime/runner.py`
  Behavior reused: run lifecycle and validation ordering.
- `investigator/runtime/validation.py`
  Behavior reused: output/evidence validation entrypoints.
- `investigator/rca/engine.py`
  Behavior reused: deterministic candidate narrowing remains pre-LLM.
- `investigator/compliance/engine.py`
  Behavior reused: deterministic evidence cataloging remains pre-LLM.
- `investigator/incident/engine.py`
  Behavior reused: deterministic representative selection remains pre-LLM.

## Context and Orientation

Right now, each engine declares `model_name`, `temperature`, and `prompt_template_hash` as class attributes, but those values are only copied into run metadata. To support real RLM behavior, model execution must become an explicit runtime step with deterministic inputs, schema-constrained outputs, and stable prompt identity.

This phase is intentionally narrow. It does not redesign engine reasoning and does not add recursive subcalls. It only creates the shared LLM invocation machinery the later phases depend on.

## Plan of Work

Create `investigator/runtime/llm_client.py` with a provider-agnostic interface, for example:

- `RuntimeModelClient.generate_structured(request: StructuredGenerationRequest) -> StructuredGenerationResult`

Add OpenAI implementation first (`OpenAIModelClient`) and keep provider dispatch separate from engine logic.

Update `investigator/runtime/contracts.py` and any dependent validation code so runtime metadata supports extensible providers. Keep serialization backwards-compatible for existing artifacts.

Create `investigator/prompts/` directory for versioned templates and schemas. Each judgment type gets:

- prompt template file,
- JSON schema file,
- deterministic hash computed from canonical bytes.

Add `investigator/runtime/llm_loop.py` with deterministic retry behavior:

- attempt 1: structured call,
- attempt 2: repair call only when output violates schema,
- hard fail with `MODEL_OUTPUT_INVALID` if still invalid.

Integrate token and cost accounting into runtime signals and run records. Cost is computed per call and aggregated by run.

Set initial budget defaults for proof mode:

- per run `max_cost_usd` default 0.25,
- full proof cap default 3.00,
- enforce stop behavior when cap is reached.

Test isolation strategy:

- unit tests use `FakeModelClient` fixture with deterministic outputs and usage,
- no unit test depends on live network,
- optional integration tests run only when API key env var is present.

## Concrete Steps

All commands run from repository root.

1. Write failing tests for runtime model client, prompt hashing, and schema-parse behavior.
   - `uv run pytest tests/unit -q -k "phase8 and llm and runtime"`
2. Implement `investigator/runtime/llm_client.py` and `investigator/runtime/llm_loop.py`.
3. Implement prompt registry and schema files under `investigator/prompts/`.
4. Update runtime contracts/runner integration for provider extensibility and cost tracking.
5. Re-run focused tests.
   - `uv run pytest tests/unit -q -k "phase8 and llm and runtime"`
6. Run broader runtime and workflow regressions.
   - `uv run pytest tests/unit -q -k "runtime or workflow"`

## Validation and Acceptance

Acceptance requires all of the following:

- At least one engine path can call shared runtime LLM loop and return schema-valid output.
- Run record usage fields show real model usage values and non-zero cost for that path.
- Parse failures produce `status=failed` with `error.code=MODEL_OUTPUT_INVALID`.
- Prompt hash in run artifacts maps to a versioned template+schema pair on disk.
- Deterministic-only runs still pass existing tests without requiring model calls.

## Idempotence and Recovery

This phase is additive. If live model calls fail in local environments, keep deterministic path enabled with an explicit runtime flag and continue testing with fake model clients. Failures must still emit run records and never skip artifact persistence.

If cost cap is reached during generation, runtime must return contract-valid partial/failure status with persisted usage snapshots.

## Artifacts and Notes

Expected new/updated files:

- `investigator/runtime/llm_client.py`
- `investigator/runtime/llm_loop.py`
- `investigator/runtime/contracts.py`
- `investigator/runtime/runner.py`
- `investigator/prompts/README.md`
- `investigator/prompts/<capability>/<template>.md`
- `investigator/prompts/<capability>/<schema>.json`
- `tests/unit/test_runtime_llm_client_phase8.py`
- `tests/unit/test_runtime_llm_loop_phase8.py`
- `tests/unit/test_runtime_prompt_registry_phase8.py`

## Interfaces and Dependencies

Required interfaces after this phase:

- `RuntimeModelClient.generate_structured(...)`
- `StructuredGenerationRequest` and `StructuredGenerationResult` types with usage/cost fields.
- Prompt registry loader that resolves `prompt_template_hash` to versioned prompt+schema artifacts.

Dependency expectations:

- OpenAI API key from environment for live integration calls.
- Unit tests use fake clients and deterministic fixtures only.

Revision Note (2026-02-10): Initial Phase 8A plan created to introduce a shared real LLM invocation loop before recursive execution or engine migration.
Revision Note (2026-02-10): Revised after Nainy review to specify provider abstraction, JSON-schema strategy, prompt versioning, explicit cost caps, and deterministic test isolation.
