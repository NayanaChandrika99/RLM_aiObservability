# Phase 11 ExecPlan: ARCgentica Procedure to Beat TRAIL GAIA with Semantic Faithfulness

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

After this work, Nainy can run one end-to-end, reproducible procedure that uses ARCgentica-style recursive REPL agents to produce TRAIL-formatted error judgments on the exact TRAIL GAIA split, score those outputs with the official TRAIL scorer, and verify semantic faithfulness constraints before accepting results. The user-visible outcome is a single runnable workflow that reports both benchmark metrics and evidence-grounded quality checks, so we optimize leaderboard performance without allowing fabricated evidence or invalid span references.

In plain language, “semantic faithfulness” here means every predicted error must point to a real span in the trace, include evidence text that can be found in trace content, and avoid unsupported claims.

## Progress

- [x] (2026-02-13 18:56Z) Confirmed user constraints for this phase: GAIA-only milestone, `gpt-5.2`, ARCgentica implementation path, optimize score and semantic faithfulness together.
- [x] (2026-02-13 18:56Z) Read TRAIL paper + benchmark code and documented exact split/protocol behavior.
- [x] (2026-02-13 18:56Z) Added canonical benchmark notes at `trail-benchmark/TRAIL_WORKING_NOTES.md`.
- [x] (2026-02-13 18:56Z) Drafted this end-to-end execution plan before implementation.
- [x] (2026-02-13 20:42Z) Added initial GAIA runner implementation in `arcgentica/` with scorer-compatible file layout and deterministic schema-valid outputs.
- [x] (2026-02-13 20:42Z) Added Phase 11 unit tests for output path conventions, one-file-per-trace generation, output schema keys, and timeout-category detection.
- [x] (2026-02-13 20:42Z) Completed a GAIA smoke generation run and verified official scorer ingestion on generated outputs.
- [x] (2026-02-13 19:22Z) Added `arcgentica/trail_semantic_checks.py` with strict semantic faithfulness enforcement, including span-location repair/drop and evidence grounding repair/drop behavior.
- [x] (2026-02-13 19:22Z) Integrated semantic checks into `arcgentica/trail_main.py` so `--semantic-checks strict` applies validation before writing output files.
- [x] (2026-02-13 19:22Z) Added semantic-check unit tests and verified strict-mode smoke output has zero invalid span references and zero ungrounded evidence by validator semantics.
- [x] (2026-02-13 19:29Z) Added semantic report artifact writing in `trail_main.py`, including run-level totals and per-trace semantic diagnostics at `arcgentica/output/trail_semantic_report.json`.
- [x] (2026-02-13 19:36Z) Implemented `arcgentica/trail_compare.py` to merge baseline metrics, candidate metrics, and semantic diagnostics into a machine-readable comparison report.
- [x] (2026-02-13 19:36Z) Added Phase 11 comparison tests for split-aware metrics discovery, metric parsing, acceptance gating, and JSON artifact persistence.
- [x] (2026-02-13 19:48Z) Attempted full GAIA official baseline run with `run_eval.py` and `openai/gpt-5.2`; execution blocked mid-run by OpenAI quota exhaustion after 34/117 outputs.
- [x] (2026-02-13 19:48Z) Scored partial baseline outputs in `results_baseline_w1` to capture current state and failure modes for reproducibility.
- [x] (2026-02-13 21:12Z) Implemented Milestone 4 recursive investigation in `arcgentica/trail_agent.py` using Agentica `spawn`/`call_agent` with chunk planning, bounded sub-agent budgets, and deterministic merge/dedupe.
- [x] (2026-02-13 21:12Z) Added `arcgentica/trail_prompt.py` and CLI budget controls in `trail_main.py`/`trail_common.py` for max agents, chunk count, chunk span size, and span text budget.
- [x] (2026-02-13 21:12Z) Ran one failure-heavy GAIA trace (`915d2c66879657f694f88e0ed6f02cf5`) with agentic mode on; confirmed multi-agent delegation (`agent-4` planner + `agent-5..8` chunk investigators) and grounded semantic report output.
- [x] Implement GAIA-only ARCgentica TRAIL runner and schema-safe output pipeline.
- [x] Implement semantic faithfulness validator and integrate it into run acceptance.
- [ ] Run baseline vs ARCgentica comparisons with official TRAIL scorer and record artifacts.
- [ ] Finalize outcomes and retrospective with measured gains and remaining gaps.

## Surprises & Discoveries

- Observation: TRAIL “split” in released code means source dataset (`GAIA` vs `SWE Bench`), not train/dev/test.
  Evidence: `trail-benchmark/benchmarking/run_eval.py` `--split` behavior and path join logic.
- Observation: Paper text and released files contain a trace-count inconsistency for GAIA in Table 5.
  Evidence: `TRAIL.pdf` table extraction indicates `118` GAIA traces while repo folders contain `117` GAIA files and total `148` files.
- Observation: One released GAIA processed annotation JSON is malformed and fails strict parsing.
  Evidence: `trail-benchmark/benchmarking/processed_annotations_gaia/a96c6811716c0473b86a23321db79c34.json`.
- Observation: Multiple GAIA traces exceed current `openai/gpt-5.2` request budgets in this environment (TPM and context-window), and then org quota was exhausted before run completion.
  Evidence: `run_eval.py` live output reported `Requested 621506/689483/861526 TPM` against `Limit 500000`, plus context window overflow (`325826` input tokens vs `272000` limit), then `You exceeded your current quota`.
- Observation: Agentica planner calls failed when prompts relied on injected additional Python resources for root planning; embedding chunk/taxonomy payloads inline in the task prompt removed that failure mode.
  Evidence: `agent-3.log` ended with runtime error about missing `chunk_catalog`/`trace_summary`; after inline prompt composition, run produced planner output and spawned chunk agents (`agent-4` through `agent-8`).

## Decision Log

- Decision: Use the exact TRAIL GAIA split from released folders as benchmark target, without introducing a custom benchmark split.
  Rationale: Nainy requested exact TRAIL split protocol.
  Date/Author: 2026-02-13 / Codex
- Decision: Implement the procedure in `arcgentica/` and only reuse ideas from `investigator/`, not the other way around.
  Rationale: Nainy requested ARCgentica as primary implementation locus.
  Date/Author: 2026-02-13 / Codex
- Decision: Primary model target for milestone execution is `openai/gpt-5.2`.
  Rationale: Nainy explicitly selected this model.
  Date/Author: 2026-02-13 / Codex
- Decision: GAIA-only is scope for this milestone; SWE Bench remains optional follow-up smoke validation.
  Rationale: User requested GAIA-only for now.
  Date/Author: 2026-02-13 / Codex
- Decision: Do not accept any “score gain” run unless semantic faithfulness checks pass.
  Rationale: User requested leaderboard optimization and faithfulness together.
  Date/Author: 2026-02-13 / Codex
- Decision: Feed root/chunk Agentica calls with inline JSON payload blocks rather than relying on injected additional Python resources.
  Rationale: Inline payloads proved more robust for this trace-debugging workload and avoided brittle runtime-variable availability errors.
  Date/Author: 2026-02-13 / Codex

## Outcomes & Retrospective

This section is intentionally provisional until implementation and evaluation complete.

Target completion outcomes:

1. A GAIA-only ARCgentica TRAIL runner exists and emits official-format JSON files per trace.
2. Official TRAIL metrics show improvement over baseline runs on the same split.
3. Semantic faithfulness checks are reported for every run and pass defined acceptance thresholds.
4. Artifacts and command transcripts are saved so results can be rerun by a novice.

## References

The following files were read fully before drafting this plan, and the listed behaviors will be reused.

- `PLANS.md`
  Behavior reused: ExecPlan structure, living-document requirements, and novice-oriented specificity.
- `AGENTS.md`
  Behavior reused: repository process constraints, complexity triage, and explicit decision recording.
- `xnotes.md`
  Behavior reused: RLM strategy preferences (recursive delegation, REPL-centric execution, context isolation).
- `trail-benchmark/TRAIL.pdf`
  Behavior reused: taxonomy, GAIA/SWE split definition, and benchmark evaluation framing.
- `trail-benchmark/benchmarking/run_eval.py`
  Behavior reused: expected output file layout and per-trace inference loop for selected split folder.
- `trail-benchmark/benchmarking/calculate_scores.py`
  Behavior reused: official metric computation pipeline and category normalization behavior.
- `trail-benchmark/TRAIL_WORKING_NOTES.md`
  Behavior reused: canonical local summary of split/protocol quirks and reproducible command set.
- `arcgentica/main.py`
  Behavior reused: async run orchestration, concurrency control, run directory conventions.
- `arcgentica/solve.py`
  Behavior reused: attempt/retry lifecycle, sandboxed code execution pattern, and attempt artifact structure.
- `arcgentica/arc_agent/agent.py`
  Behavior reused: recursive `call_agent` pattern, scoped object passing, and bounded sub-agent counting.
- `arcgentica/arc_agent/prompts.py`
  Behavior reused: prompt structure for hypothesis exploration and delegated sub-task framing.
- `arcgentica/common.py`
  Behavior reused: dataclass-style config/result serialization patterns.
- `arcgentica/score.py`
  Behavior reused: deterministic evaluation output handling and scoring utility conventions.
- `investigator/runtime/repl_loop.py`
  Behavior reused: budget-aware iterative REPL loop semantics and submit-deadline guardrails.
- `investigator/runtime/repl_interpreter.py`
  Behavior reused: bounded helper APIs (`call_tool`, `llm_query`) and usage metering patterns.
- `investigator/rca/engine.py`
  Behavior reused: evidence deduplication, deterministic fallback philosophy, and runtime signal tracking.

## Context and Orientation

TRAIL is a trace-debugging benchmark where each input file is a full structured execution trace and each gold file lists multiple labeled errors with location pointers (`span_id`). ARCgentica is currently built for ARC grid reasoning, but its core strength is a recursive, stateful REPL agent runtime that can be repurposed to trace-debugging tasks.

For this phase, we treat GAIA as one fixed benchmark folder:

- Inputs: `trail-benchmark/benchmarking/data/GAIA/*.json`
- Gold: `trail-benchmark/benchmarking/processed_annotations_gaia/*.json`

The official scorer expects one generated file per trace filename. We must preserve that format exactly. We also keep the TRAIL taxonomy leaf categories exactly as defined by the benchmark prompt, because scorer normalization is permissive but not perfect.

This plan defines a runner procedure that balances two goals:

1. Maximize official GAIA benchmark metrics.
2. Enforce semantic faithfulness checks so outputs remain evidence-grounded.

## Plan of Work

The work proceeds in seven milestones.

Milestone 1 creates a TRAIL-specific runner in `arcgentica/` without changing TRAIL benchmark code. The runner reads GAIA trace JSON files, executes ARCgentica-style recursive analysis, and writes one output JSON per trace in official format.

Milestone 2 introduces a TRAIL schema module and post-processor. This module normalizes categories to official leaf names, deduplicates `(location, category)` pairs, validates required score fields, and guarantees strict JSON-only output files.

Milestone 3 adds semantic faithfulness tooling. For each predicted error, the validator checks that location exists in the trace and that evidence text is grounded in extracted span/message/tool text. Predictions failing checks are either repaired (if deterministic repair is possible) or dropped with a recorded gap in run diagnostics.

Milestone 4 adds recursive investigation strategy tuned for TRAIL traces. The parent agent performs coarse narrowing over span families, delegates focused subcalls for candidate categories, and merges results by evidence strength and confidence. This must remain budget-bounded and deterministic in ordering.

Milestone 5 adds scoring and reporting integration. The pipeline runs the official TRAIL scorer unchanged and then runs semantic quality checks, producing one combined report per run.

Milestone 6 performs baseline versus ARCgentica comparisons on GAIA and records all outputs under a dedicated artifact directory.

Milestone 7 finalizes documentation, outcomes, and rerun instructions so a novice can reproduce the same benchmark and quality checks.

## Concrete Steps

All commands below run from repository root unless explicitly noted.

1. Prepare environment and baseline outputs.

    cd trail-benchmark/benchmarking
    python3 run_eval.py --model openai/gpt-5.2 --data_dir data --output_dir results_baseline --split GAIA --max_workers 5
    python3 calculate_scores.py --results_dir results_baseline

Expected observation: one `outputs_* -GAIA` directory and corresponding metrics text output from official scorer.

2. Implement ARCgentica TRAIL runner modules (Milestones 1-4) under `arcgentica/`.

Planned file additions/updates:

- `arcgentica/trail_main.py` (CLI entrypoint)
- `arcgentica/trail_common.py` (config, dataclasses, paths)
- `arcgentica/trail_agent.py` (recursive agent wrapper)
- `arcgentica/trail_prompt.py` (TRAIL taxonomy prompt templates and constraints)
- `arcgentica/trail_semantic_checks.py` (faithfulness validation)
- `arcgentica/pyproject.toml` (only if dependency additions are required)

3. Run ARCgentica GAIA generation with semantic checks enabled.

    cd arcgentica
    uv run python trail_main.py \
      --trail-data-dir ../trail-benchmark/benchmarking/data \
      --split GAIA \
      --model openai/gpt-5.2 \
      --output-dir ../trail-benchmark/benchmarking/results_arcgentica \
      --semantic-checks strict

Expected observation: generated files in a scorer-compatible output directory and a semantic-check report artifact.

4. Score ARCgentica output using official TRAIL scorer.

    cd ../trail-benchmark/benchmarking
    python3 calculate_scores.py --results_dir results_arcgentica

Expected observation: weighted F1, location accuracy, and joint accuracy printed and written to `*-metrics.txt`.

5. Produce comparison report (baseline vs ARCgentica + faithfulness summary).

    cd ../..
    uv run python arcgentica/trail_compare.py \
      --baseline-dir trail-benchmark/benchmarking/results_baseline \
      --candidate-dir trail-benchmark/benchmarking/results_arcgentica \
      --semantic-report arcgentica/output/trail_semantic_report.json \
      --out artifacts/evaluation/trail_gaia_arcgentica_report.json

Expected observation: one machine-readable report and one concise terminal summary.

## Validation and Acceptance

Acceptance requires all conditions below.

1. Protocol correctness:
   - Uses exact TRAIL GAIA split folder, no alternative benchmark partition.
   - Uses official TRAIL scorer without modification.
2. Output correctness:
   - Every generated file is parseable JSON and matches required TRAIL output schema keys.
3. Semantic faithfulness:
   - 100% of predicted locations resolve to real span IDs in the corresponding trace.
   - Evidence-grounding checker passes configured threshold (default target: >= 0.95 grounded evidence rate).
   - Any dropped/invalid predictions are logged in diagnostics.
4. Benchmark improvement:
   - Candidate run exceeds baseline run on GAIA joint accuracy.
   - Candidate run must beat published GAIA joint reference of 0.18 from TRAIL paper.
5. Reproducibility:
   - Commands in this plan rerun end-to-end and regenerate reports without manual intervention.

## Idempotence and Recovery

Runner outputs must be written to new run directories by timestamp or run ID so reruns do not overwrite prior results. If a run fails mid-way, rerunning with the same run ID should skip already completed trace files unless `--force` is provided.

If semantic checks fail globally, do not discard outputs; write failure diagnostics and keep artifacts for inspection. If scorer fails due to malformed upstream gold files, record the exact filename and continue with remaining files while reporting adjusted denominator behavior.

No destructive operations are part of this plan.

## Artifacts and Notes

Required artifacts after execution:

- `trail-benchmark/benchmarking/results_baseline/...`
- `trail-benchmark/benchmarking/results_arcgentica/...`
- `arcgentica/output/trail_semantic_report.json`
- `artifacts/evaluation/trail_gaia_arcgentica_report.json`
- Updated notes if new benchmark quirks are discovered:
  `trail-benchmark/TRAIL_WORKING_NOTES.md`

Useful debugging notes should include:

- run command and model id
- file-level generation failures
- semantic check failure counts by reason
- scorer output summary

## Interfaces and Dependencies

The final implementation must expose:

- A CLI entrypoint for GAIA generation (`trail_main.py`) with explicit flags for model, split, data path, output path, and semantic check mode.
- A pure function that validates one generated prediction file against one raw trace for semantic faithfulness.
- A report writer that merges official scorer outputs with semantic validation outputs.

Dependencies should remain minimal and align with existing ARCgentica toolchain (`uv`, `symbolica-agentica`, and current Python runtime). Any new dependency must be justified by direct need for parsing, validation, or reproducibility.

Revision Note (2026-02-13): Initial Phase 11 ExecPlan created because Nainy requested an execfile first for the full GAIA procedure before implementation.
