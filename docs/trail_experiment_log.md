# TRAIL Experiment Log

All experiments run on the **dev18 subset** (18 GAIA traces covering all 21 error categories).

**Baseline reference** (GPT-5.2, full 117 GAIA traces, official eval): F1=0.438, Location=0.478, Joint=0.186

---

## Results Summary

| # | Experiment | Mode | F1 | Location | Joint | Notes |
|---|-----------|------|-----|----------|-------|-------|
| 1 | smoke_heuristic | heuristic | 0.053 | 0.007 | 0.000 | Regex-only baseline, no LLM |
| 2 | live_check | single_pass | 0.226 | 0.054 | 0.004 | First live single-pass run |
| 3 | live_check_fix1 | single_pass | 0.237 | 0.096 | 0.060 | JSON parsing fixes |
| 4 | live_check_fix2 | single_pass | 0.225 | 0.110 | 0.032 | Further parsing fixes |
| 5 | live_check_fix3 | single_pass | 0.264 | 0.068 | 0.023 | Prompt tweaks |
| 6 | live_check_fix4 | single_pass | 0.130 | 0.000 | 0.000 | Regression — context/retry fix was harmful |
| 7 | postfix_01 | single_pass | 0.053 | 0.007 | 0.000 | Fell back to heuristic (single_pass failed) |
| 8 | repl_01 | REPL agentic | 0.260 | 0.197 | 0.009 | First agentic REPL run |
| 9 | repl_03_true_mw3 | REPL agentic | 0.196 | 0.108 | 0.009 | max_workers=3 |
| 10 | repl_04_guardrail_resume | REPL agentic | 0.257 | 0.139 | **0.111** | Best joint accuracy |
| 11 | repl_05_location_refine | REPL agentic | 0.225 | 0.128 | 0.014 | Added location refinement step |
| 12 | repl_06_health_noresume | REPL agentic | **0.311** | **0.280** | 0.015 | Best F1 and Location |
| 13 | repl_07_desc_signal | REPL agentic | 0.240 | 0.097 | 0.056 | Description-based signal scoring |
| 14 | repl_08_revert_r1b | REPL agentic | 0.262 | 0.083 | 0.004 | Revert experiment, run 1 |
| 15 | repl_08_revert_r2b | REPL agentic | 0.236 | 0.028 | 0.000 | Revert experiment, run 2 |
| 16 | repl_08_revert_r3b | REPL agentic | 0.208 | 0.021 | 0.006 | Revert experiment, run 3 |
| 17 | repl_09_kw_anchor | REPL agentic | 0.210 | 0.042 | 0.000 | Keyword anchor for location |
| 18 | revert_clean_r1 | REPL agentic | 0.267 | 0.047 | 0.004 | Clean revert, run 1 |
| 19 | revert_clean_r2 | REPL agentic | 0.239 | 0.125 | 0.056 | Clean revert, run 2 |
| 20 | revert_clean_r3 | REPL agentic | 0.204 | 0.042 | 0.000 | Clean revert, run 3 |
| 21 | precision_gate_r1 | REPL agentic | 0.113 | 0.000 | 0.000 | Precision gating — too aggressive, killed recall |
| 22 | precision_gate_r2 | REPL agentic | 0.254 | 0.083 | 0.006 | Relaxed precision gate |
| 23 | adjudicate_r1 | REPL agentic | 0.254 | 0.083 | 0.006 | Adjudication step added |
| 24 | refine_delta1_r1 | REPL agentic | 0.168 | 0.035 | 0.006 | Delta refinement — regressed |
| 25 | localserver_fix_r1 | REPL agentic | 0.229 | 0.153 | 0.024 | Local server env fix |
| 26 | pairloc_litellm_bias_r1 | REPL agentic | 0.229 | 0.153 | 0.024 | LiteLLM span name bias for location |
| 27 | pairloc_catspan_bias_r1 | REPL agentic | 0.247 | 0.216 | 0.024 | Category-span affinity bias |

All experiments use **openai/gpt-5-mini** and **prompt v2**.

---

## Key Observations

### Best Results

| Metric | Best Experiment | Score |
|--------|----------------|-------|
| Weighted F1 | repl_06_health_noresume (#12) | 0.311 |
| Location Accuracy | repl_06_health_noresume (#12) | 0.280 |
| Joint Accuracy | repl_04_guardrail_resume (#10) | 0.111 |

### Run-to-Run Variance

The revert experiments (r1b/r2b/r3b and clean r1/r2/r3) reveal significant non-determinism:

| Metric | Min | Max | Range |
|--------|-----|-----|-------|
| F1 (revert_r1-3) | 0.208 | 0.262 | 0.054 |
| Location (revert_r1-3) | 0.021 | 0.083 | 0.062 |
| F1 (clean_r1-3) | 0.204 | 0.267 | 0.063 |
| Location (clean_r1-3) | 0.042 | 0.125 | 0.083 |

This ~0.06 F1 / ~0.08 Location variance means small improvements may be noise.

### Mode Comparison

| Mode | Experiments | Avg F1 | Avg Location | Avg Joint |
|------|-----------|--------|-------------|-----------|
| Heuristic | 1 | 0.053 | 0.007 | 0.000 |
| Single-pass | 5 (excl. regressions) | 0.238 | 0.082 | 0.030 |
| REPL Agentic | 20 | 0.230 | 0.105 | 0.020 |

REPL agentic mode has slightly better Location accuracy on average but similar F1 to single-pass. Both dramatically outperform heuristic.

### What Improved Scores

1. **Health check + no resume (#12)**: Best overall — delegation health monitoring prevented silent failures
2. **Guardrail resume (#10)**: Best joint — checkpoint resume prevented re-processing divergence
3. **Category-span bias (#27)**: Best recent Location — weighting span names by category affinity

### What Hurt Scores

1. **Precision gating (#21)**: F1 collapsed to 0.113 — too aggressive filtering killed recall
2. **Refine delta1 (#24)**: F1 dropped to 0.168 — delta refinement introduced noise
3. **live_check_fix4 (#6)**: Location hit 0.000 — context retry logic was broken
4. **postfix_01 (#7)**: Fell back to heuristic — single_pass silently failed

### Gap to Baseline

Our best results vs. GPT-5.2 baseline (full GAIA):

| Metric | Our Best | Baseline | Gap |
|--------|---------|----------|-----|
| F1 | 0.311 | 0.438 | -0.127 |
| Location | 0.280 | 0.478 | -0.198 |
| Joint | 0.111 | 0.186 | -0.075 |

Note: Our runs use dev18 (18 traces) while baseline uses full GAIA (117 traces). Results may not be directly comparable, but the gap is substantial.

### Root Causes of the Gap

1. **Model**: All our experiments use gpt-5-mini. Baseline uses gpt-5.2 (larger, more capable).
2. **Context budget**: Our single-pass caps GPT-5 at 20K chars. Baseline sends the full raw trace.
3. **Temperature**: Our GPT-5 models don't set temperature=0. Baseline does.
4. **Chunking fragmentation**: REPL mode breaks traces into chunks, losing global context needed for Goal Deviation, Task Orchestration, Resource Abuse (all at 0% F1).

---

## Experiment Timeline

```
Feb 13 (early)    smoke_heuristic          → Heuristic baseline
Feb 13 (18:07)    live_check               → First live single-pass
Feb 13 (evening)  live_check_fix1-4        → Iterating on parsing + prompt
Feb 13 (late)     postfix_01               → Single-pass fallback failure
Feb 13-14         repl_01 → repl_09        → Agentic REPL experiments
Feb 14 (night)    revert_clean_r1-3        → Stability/variance testing
Feb 14 (morning)  precision_gate_r1-2      → Precision filtering attempt
Feb 14 (morning)  adjudicate_r1            → Adjudication step
Feb 14 (01:03)    refine_delta1_r1         → Delta refinement (regressed)
Feb 14 (morning)  localserver_fix_r1       → Env fix
Feb 14 (morning)  pairloc_litellm_bias_r1  → Span name bias
Feb 14 (10:11)    pairloc_catspan_bias_r1  → Category-span affinity (latest)
```

---

## Phase A Changes (In Progress)

Based on the analysis above, Phase A targets the three root causes of the gap:

1. **Remove 20K char budget cap for GPT-5.2** → let model see full trace
2. **Set temperature=0 for GPT-5.2** → match baseline determinism
3. **Raise general budget ceiling from 40K to 400K** → let all models use more context

These changes are in `arcgentica/trail_agent.py`. Next step: run an experiment with GPT-5.2 in single-pass mode with these changes applied.
