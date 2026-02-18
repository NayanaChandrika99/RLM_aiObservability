# TRAIL Improvement Strategy (REPL-First)

## Executive Summary

Goal: beat TRAIL with the **Agentic REPL** architecture in `arcgentica`, not by switching the main path to single-pass.

From `arcgentica/output/experiments/experiment_log.json` and `docs/trail_experiment_log.md`:
- Best observed REPL F1: **0.311** (`exp_dev18_repl_07_desc_signal`)
- Best observed REPL Location: **0.280** (`exp_dev18_repl_07_desc_signal`)
- Best observed REPL Joint: **0.111** (`exp_dev18_repl_06_health_noresume`)

Reference baseline (official, full GAIA 117, GPT-5.2): **F1=0.438, Location=0.478, Joint=0.186**.

Primary objective: push REPL Joint to **>=0.20** with stable reliability (`analysis_fallbacks=0`, `delegation_failures=0`).

---

## 1. Scope And Non-Goals

### In scope
- Improve TRAIL accuracy through REPL planning + delegated analysis + deterministic post-processing.
- Use split model routing where needed:
  - Root planning: `openai/gpt-5.2`
  - Delegated chunk calls: `openai/gpt-5-mini`

### Out of scope
- Making single-pass the production/default strategy.
- Using answer-key style signals (`true_answer`) for category assignment.

Single-pass remains useful only as:
1. A diagnostic oracle.
2. A control baseline to identify missing REPL signals.

---

## 2. What The Experiment Log Tells Us

## 2.1 Reliability is now measurable and enforceable

Recent runs include `semantic_report.json` with:
- `analysis_fallbacks`
- `delegation_failures`

This must be used as a hard quality gate. Accuracy numbers are not trusted when these counters are non-zero.

## 2.2 Variance is high

Observed repeat variance on similar settings is substantial (~0.05-0.08 range on F1/Location in some groups), so single-run deltas are weak evidence.

Policy:
- Any “improvement” requires repeated runs and mean/std reporting before acceptance.

## 2.3 Joint bottleneck is pair precision, not only location hit

Pattern from recent REPL outputs:
- Location can increase while Joint stays flat.
- Over-prediction in categories like `Tool-related` and `Tool Definition Issues` hurts exact `(location, category)` matching.
- High-support misses in `Formatting Errors`, `Goal Deviation`, `Instruction Non-compliance` depress joint recall.

Conclusion:
- Optimize exact pair precision/recall, not raw location hit rate alone.

---

## 3. REPL-First Strategy

## Phase R0: Reliability Gates (must pass first)

Acceptance criteria for every tuning run:
- `analysis_fallbacks = 0`
- `delegation_failures = 0`
- No silent chunk drop behavior

Execution policy:
- Use `--max-workers 1` during tuning.
- Use resume checkpoints for long runs.
- Block metric comparisons when reliability gates fail.

## Phase R1: Split-Model REPL Control

Objective:
- Keep REPL architecture while improving root planning quality.

Configuration:
- Root: `openai/gpt-5.2`
- Chunk: `openai/gpt-5-mini`

Protocol:
1. Run 2-3 dev18 repeats on identical config.
2. Report mean/std for F1, Location, Joint.
3. Compare against mini-only REPL control.

Decision rule:
- Keep split-model only if Joint mean improves beyond noise band.

## Phase R2: Pair Precision Engine (joint-critical)

Objective:
- Increase exact `(location, category)` matches.

Changes:
1. Category-conditioned candidate span shortlist before final location write.
2. Stronger suppression of weak pair assignments:
   - penalize generic `Step *` spans unless evidence is direct.
   - require minimum evidence overlap for category-sensitive labels.
3. Category confusion dampening:
   - reduce easy drift into `Tool-related` and `Tool Definition Issues` when stronger categories exist.

Acceptance criteria:
- Pair precision up.
- Pair recall flat or up.
- Joint mean up across repeats.

## Phase R3: Category Recovery For Joint Recall

Targeted categories from historical misses:
- `Formatting Errors`
- `Goal Deviation`
- `Instruction Non-compliance`
- `Task Orchestration`
- `Resource Abuse`

Approach:
- Add deterministic trace-derived hints only (no label leakage).
- Keep hints as soft evidence, never hard label override without span support.

Acceptance criteria:
- Reduced FN counts in targeted categories.
- No precision collapse from broad hinting.

## Phase R4: Scale Validation

After dev18 gains stabilize:
1. Evaluate on a held-out GAIA slice not used for tuning.
2. Then run full 117-trace GAIA.
3. Report:
   - aggregate metrics
   - reliability counters
   - per-category precision/recall
   - pair precision/recall

---

## 4. Implementation Priority

| Priority | Change | Why | Acceptance |
|---|---|---|---|
| P0 | Reliability gates in every run report | Prevent false wins | `fallbacks=0`, `deleg_failures=0` |
| P0 | Split-model REPL (`5.2` root, `5-mini` chunks) | Improve planning without full-cost chunks | Joint mean beats mini-only |
| P1 | Pair-precision location/category coupling | Directly targets joint bottleneck | Pair precision up, joint up |
| P1 | Step-span suppression + evidence thresholding | Reduce false pair assignments | FP down in error-prone categories |
| P2 | Targeted category recovery hints | Improve joint recall where support is high | FN down, no large FP jump |
| P2 | Held-out + full GAIA validation | Confirm generalization | Stable gains beyond dev18 |

---

## 5. Metrics Framework (required for each ablation)

Core:
- Weighted F1
- Location Accuracy
- Joint Accuracy

Reliability:
- `analysis_fallbacks`
- `delegation_failures`

Pair diagnostics:
- Exact pair precision = `TP_pairs / Pred_pairs`
- Exact pair recall = `TP_pairs / Gold_pairs`
- Top pair FN/FP categories

Run quality:
- Mean/std across repeats
- Number of traces completed

---

## 6. Guardrails

1. No silent regressions:
   - If joint rises but reliability counters fail, reject run.
2. No leakage:
   - Do not use `true_answer` or any external gold-only signal at inference time.
3. No one-off conclusions:
   - Minimum 2 repeats for any architecture/prompt claim.
4. Keep REPL primary:
   - Single-pass results can inform hypotheses, but not replace the target architecture.

---

## 7. Immediate Next Action

Run controlled REPL split-model repeats, then tune exact pair precision:

1. `5.2/mini` repeat block (2-3 runs) under reliability gates.
2. Apply one pair-precision change at a time.
3. Re-run and compare mean/std + pair diagnostics before stacking changes.
