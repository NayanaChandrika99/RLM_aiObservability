# TRAIL SOTA Experimentation Design

**Date**: 2026-02-13
**Author**: Nainy + Claude
**Status**: Draft for approval
**Branch**: `wip/phase11-trail-gaia`

## 1. Goal

Build an experimentation framework that enables rapid iteration toward SOTA on the TRAIL GAIA benchmark using the ARCgentica Agentic REPL, starting with an enhanced single-pass prompt (Approach A) and graduating to a multi-pass agentic architecture (Approach B) when the prompt ceiling is reached.

**Target model for dev iteration**: `openai/gpt-5-mini` (fast, cheap).
**Target model for final SOTA run**: `openai/gpt-5.2` (or best available).

## 2. Baseline Numbers (the bar to beat)

GPT-5.2 on full GAIA split (117 traces, 34 completed before quota exhaustion):

| Metric | Score |
|--------|-------|
| Weighted F1 | 0.4380 |
| Location Accuracy | 0.4782 |
| Joint Accuracy | 0.1856 |
| Reliability Correlation | 0.7728 |
| Instruction Adherence Corr. | 0.3953 |
| Overall Score Correlation | 0.6896 |

Per-category F1 (0% categories = biggest opportunities):

| Category | F1 | Support |
|----------|-----|---------|
| Resource Exhaustion | 1.000 | 1 |
| Tool Definition Issues | 0.667 | 2 |
| Language-only | 0.621 | 11 |
| Instruction Non-compliance | 0.619 | 17 |
| Resource Abuse | 0.667 | 5 |
| Goal Deviation | 0.539 | 16 |
| Poor Information Retrieval | 0.500 | 5 |
| Tool Selection Errors | 0.500 | 8 |
| Formatting Errors | 0.500 | 10 |
| Incorrect Problem Identification | 0.333 | 2 |
| **Tool-related** | **0.000** | **7** |
| **Environment Setup Errors** | **0.000** | **2** |
| **Authentication Errors** | **0.000** | **2** |
| **Timeout Issues** | **0.000** | **1** |
| **Context Handling Failures** | **0.000** | **3** |
| **Task Orchestration** | **0.000** | **8** |

## 3. Scoring Mechanism Deep-Dive

Understanding how the official scorer works is critical for optimization.

### 3.1 Weighted F1 (multi-label binary)

For each trace, the scorer creates a 21-dimensional binary vector: `y[i] = 1` if category `i` appears in the error list, else 0. F1 is computed over these binary vectors aggregated across all traces using `sklearn.metrics.f1_score(average='weighted')`.

**Implication**: Having 3 errors of the same category = same score as having 1. The scorer cares about *which categories are present per trace*, not how many errors of each.

### 3.2 Location Accuracy (set intersection)

Per-trace: `|gold_locations ∩ pred_locations| / |gold_locations|`, averaged across traces.

**Implication**: This is a set operation over span_ids. Duplicate locations don't help. Missing even one gold span_id reduces the ratio. The scorer does NOT do bipartite matching -- it's raw set intersection.

### 3.3 Joint Accuracy (the hardest metric)

Per-trace: `|gold_(loc,cat) pairs ∩ pred_(loc,cat) pairs| / |gold_(loc,cat) pairs|`, averaged.

**Implication**: Need BOTH the right category AND the right span_id. This is where the baseline fails most (0.186 vs 0.438 F1), confirming that **location accuracy is the primary bottleneck**, not category classification.

### 3.4 Score Correlations

Pearson r between gold and predicted rubric scores. Currently computed via formula-based `_score_block()` which uses `max(1, 5 - error_count)`. This is almost certainly hurting correlation, especially for instruction adherence (r=0.39).

### 3.5 Category Normalization

The scorer normalizes categories via case-insensitive matching and substring matching (`normalize_category()`). This means small naming variations are tolerated, but the model should still use exact leaf names.

### 3.6 First/Last Occurrence Rule

From the official prompt: "In the case of 'Resource Abuse' error, only mark the **last** instance. For all other errors, mark the **first** instance."

This rule is critical for location accuracy and is currently NOT enforced in the ARCgentica prompts.

## 4. Gold Annotation Analysis

### 4.1 Category Distribution (across 117 GAIA traces)

| Category | Traces Containing | Est. Instances |
|----------|-------------------|----------------|
| Formatting Errors | ~75 | ~150+ |
| Goal Deviation | ~70 | ~75 |
| Instruction Non-compliance | ~65 | ~75 |
| Tool Selection Errors | ~50 | ~55 |
| Tool-related | ~45 | ~55 |
| Language-only | ~40 | ~45 |
| Task Orchestration | ~40 | ~50 |
| Resource Abuse | ~30 | ~35 |
| Poor Information Retrieval | ~25 | ~30 |
| Context Handling Failures | ~20 | ~22 |
| Incorrect Problem Identification | ~15 | ~18 |
| Tool Output Misinterpretation | ~10 | ~12 |
| Environment Setup Errors | ~8 | ~9 |
| Resource Not Found | ~7 | ~8 |
| Authentication Errors | ~5 | ~6 |
| Resource Exhaustion | ~3 | ~3 |
| Timeout Issues | ~3 | ~3 |
| Service Errors | ~2 | ~2 |
| Tool Definition Issues | ~1 | ~1 |

### 4.2 Errors Per Trace

- Min: 1, Max: ~15, Mean: ~4.5, Median: ~4
- **Every trace has at least 1 error** (no zero-error traces)
- ~15 traces have exactly 1 error (usually LOW-impact Formatting Errors)
- ~52 traces have 5+ errors

### 4.3 Impact Distribution

- HIGH: ~44%, MEDIUM: ~29%, LOW: ~27%

## 5. Why the Baseline Fails -- Root Cause Analysis

### 5.1 Location errors (verified on trace `0242ca2533...`)

The baseline gets 4/5 **categories** correct but only 1/5 **locations** correct for this trace. The model points to a parent span or sibling span instead of the exact span where the error first manifests. The first/last occurrence rule is insufficiently reinforced in the baseline prompt.

### 5.2 Hard category detection patterns

Analysis of gold annotations for the 0% F1 categories reveals they all require **structural trace analysis** -- examining the relationship between spans -- not just reading LLM output text:

**Tool-related (7 traces, 0% F1)**:
- Signal: LLM span output claims "I have verified..." or "According to the USGS record..." but no preceding TOOL span exists with a matching `tool.name`. The agent hallucinated a tool interaction.
- Detection: Compare plan's required tool calls against actual TOOL spans present in trace.

**Context Handling Failures (3 traces, 0% F1)**:
- Signal: `Address:` field in tool response shows agent is on page X, but the next action only makes sense for page Y. Or: agent receives error feedback but next action repeats the same mistake.
- Detection: Correlate `Address` fields across sequential tool responses; check if error messages flow into subsequent LLM behavior.

**Task Orchestration (8 traces, 0% F1)**:
- Signal: Agent writes a detailed 7-step plan, then immediately calls `final_answer` without executing any intermediate tool calls. Low ratio of actual TOOL spans to planned steps.
- Detection: Count distinct `tool.name` values in child TOOL spans vs tool calls mentioned in plan text.

**Environment Setup Errors (2 traces, 0% F1)**:
- Signal: TOOL spans with `status_code: "Error"` where `status_message` contains `FileNotFoundError`, `InterpreterError` (forbidden ops like `open()`), or `UnboundLocalError` (tool internal bug).
- Detection: Check `exception.type` and `status_message` in TOOL/execution spans for infrastructure-level errors.

### 5.3 Score correlation weakness

The formula-based `_score_block()` uses `max(1, 5 - error_count)` for all dimensions. Gold annotations have nuanced scores: a trace can have many errors but still score 4/5 on instruction adherence if the errors are system-level. The formula cannot capture this nuance.

## 6. Approach A: Enhanced Single-Pass Prompt

### 6.1 Architecture

Same as baseline: full trace dumped into a single LLM call. But with a dramatically improved prompt and post-processing.

### 6.2 Prompt Improvements (versioned as `trail_prompt_v2.py`)

**A1. Trace Structure Guide** (prepended):
```
Before analyzing, understand the trace structure:
- Each span has a unique `span_id` -- use these exact IDs for error locations
- `child_spans` contain nested execution steps within a parent span
- `logs[].body` contains function names, arguments, and outputs
- Look at `status_code` and `status_message` for execution errors
- `span_name` identifies the operation type (e.g., "LiteLLMModel.__call__", "FinalAnswerTool")
- TOOL spans have `tool.name` in their attributes or `span_name`
```

**A2. Category-Specific Detection Guidance** (embedded in taxonomy):

For each leaf category, add a detection hint explaining what signal to look for in the trace. Examples:

```
Tool-related (fabricating tool outputs/capabilities):
  DETECT: The LLM output text claims a tool returned specific information
  (e.g., "According to the search results..."), but no TOOL span with
  that tool.name exists as a child/sibling span. Compare stated tool
  usage in LLM output against actual TOOL spans in the trace.

Context Handling Failures:
  DETECT: The agent performs an action appropriate for page/context A,
  but the Address/state field in the preceding tool response shows it
  is actually on page/context B. Or: the agent receives an error message
  but the next action repeats the same or trivially-different mistake.

Task Orchestration:
  DETECT: A multi-step plan is written but the only TOOL child span is
  final_answer -- no intermediate tools were actually called. Or: the
  agent says "I will now use tool X" but no X span follows.

Environment Setup Errors:
  DETECT: TOOL spans with status_code="Error" containing FileNotFoundError,
  InterpreterError ("not permitted to evaluate"), or UnboundLocalError
  in an internal tool path.
```

**A3. Location Precision Instructions**:
```
CRITICAL LOCATION RULES:
- For ALL errors except "Resource Abuse": use the span_id where the error FIRST appears
- For "Resource Abuse" ONLY: use the span_id of the LAST instance
- The location MUST be an exact span_id from the trace (e.g., "3bcc157b63d51414")
- Prefer the most specific (deepest child) span where the error manifests
- Do NOT use a parent span when a child span is more precise
```

**A4. Two Diverse Few-Shot Examples**:
- Example 1: Reasoning errors (Tool-related hallucination + Goal Deviation + Instruction Non-compliance)
- Example 2: System/planning errors (Environment Setup Errors + Task Orchestration + Context Handling Failures)

Each example shows the exact span_id selection rationale.

**A5. LLM-Generated Scores**:
Replace the formula-based scoring template with explicit instructions to reason about each dimension independently based on the specific errors found. Include guidance like:
```
- reliability_score: How reliable were the system's outputs? Factor in
  hallucinations and incorrect information more heavily than formatting issues.
- security_score: Only reduce from 5 if there are Authentication Errors,
  data exposure risks, or unsafe API calls.
- instruction_adherence_score: Consider whether the system followed the
  task instructions. Missing end_plan tags are LOW impact; ignoring the
  core task requirement is HIGH impact.
- plan_opt_score: Was the plan efficient? Redundant tool calls, failed
  retries without adaptation, and excessive resource usage reduce this.
```

**A6. Smart Trace Truncation** (for context window overflow):
When the trace exceeds the model's context limit:
1. Parse all span_ids and build a compact span index (id, name, status_code, parent)
2. Include full text for spans with error signals (errors, failures, exceptions, status_code != "Unset")
3. Truncate normal spans to first 200 chars of their logs
4. Always include the root span and final answer span in full
5. Append the complete span_id index at the end so the model can reference any span

### 6.3 Post-Processing

- Validate all `location` values are real span_ids from the trace
- Normalize categories via the same logic as the official scorer
- Deduplicate (location, category) pairs
- Apply first/last occurrence correction for multi-instance errors

## 7. Approach B: Two-Pass Agentic (graduation from A)

Activated when Approach A hits a ceiling or >20% of traces fail due to context window overflow.

### 7.1 Architecture

```
Trace JSON
    |
    v
[Pass 1: Trace Profiler Agent]
    - Sees: full trace (or smart truncation)
    - Produces: {
        task_description: str,
        execution_flow: [span_id, span_name, status]...,
        suspicious_spans: [{span_id, reason}...],
        span_index: [{span_id, name, parent, has_error}...]
      }
    - Budget: ~4K output tokens, 1 agent
    |
    v
[Pass 2: Error Investigators] (parallel, max 4 agents)
    - Each receives: trace profile + assigned span group + full taxonomy with detection hints
    - Each produces: error findings with exact span_ids + evidence
    - Budget: ~2K output tokens per agent
    |
    v
[Pass 3: Synthesizer Agent]
    - Receives: all findings + trace profile + full taxonomy
    - Produces: {
        errors: deduplicated + first/last occurrence applied,
        scores: LLM-reasoned rubric scores
      }
    - Budget: ~3K output tokens, 1 agent
```

### 7.2 Key Improvement Over Milestone 4

The current milestone 4 implementation has chunk agents that are **isolated** -- they only see their assigned spans. This prevents detecting errors that require global context (Goal Deviation, Task Orchestration). The two-pass design fixes this by:
1. Pass 1 extracts a **trace profile** that provides global context
2. Pass 2 agents receive this profile along with their spans
3. Pass 3 can cross-reference findings across agents

## 8. Dev Subset

18 traces selected to cover all 21 categories with a balanced mix:

### Easy (1-2 errors, common categories):
1. `0adc4f3b99d9564d32811e913cc9d248` -- 1 error: Formatting Errors (LOW)
2. `27a6c5ebc3311542156fdde857a0035f` -- 1 error: Formatting Errors (LOW)
3. `99f6b447779ba86b3cff2caede832d59` -- 1 error: Instruction Non-compliance (LOW)
4. `72877db591837666d500b459fb3cf29d` -- 2 errors: Instruction Non-compliance (LOW x2)
5. `41b597524173272503073a0799ac523c` -- 1 error: Instruction Non-compliance (HIGH)

### Medium (3-5 errors, mixed categories):
6. `0035f455b3ff2295167a844f04d85d34` -- 3 errors: Instruction Non-compliance, Tool-related, Goal Deviation
7. `3637c5845140adbf29c565923c20e94d` -- 4 errors: Tool Selection, Task Orchestration, Tool-related, Goal Deviation
8. `6e64c2e543327a7c2f2ce3d26ced94d1` -- 3 errors: Instruction Non-compliance, Tool-related, Goal Deviation
9. `3205fa0cb2135fe671bf7cd0e5a26151` -- 3 errors: Tool Selection x2, Goal Deviation
10. `18efa24e637b9423f34180d1f2041d3e` -- 4 errors: Incorrect Problem ID, Goal Deviation, Auth Errors, Context Handling

### Hard (many errors, rare categories):
11. `0140b3f657eddf76ca82f72c49ac8e58` -- 8 errors: Context Handling, Language-only, Formatting x5, Resource Abuse
12. `876eb108c8650d4ada63a8d39aa1e96c` -- 11 errors: max category coverage (Goal Dev, Task Orch, Instr NC, Formatting, Context Handling, Tool Selection, Incorrect Problem ID, Tool Output Misinterpretation)
13. `860f9d45f2e50bfecb190bb26eff1f32` -- 10 errors: Formatting, Resource Abuse, Goal Dev, Tool-related, Environment Setup, Resource Not Found, Instr NC
14. `7bf0addde339e4cac9dd3b772232a7e0` -- 8 errors: Environment Setup, Task Orch, Instr NC, Language-only, Formatting, Incorrect Problem ID, Timeout Issues, Resource Exhaustion
15. `59365b27641e501d105b0e8f5e7c5af7` -- 12 errors: Formatting x9, Tool-related, Poor Info Retrieval, Context Handling, Resource Abuse
16. `915d2c66879657f694f88e0ed6f02cf5` -- 12 errors: Formatting, Poor Info Retrieval, Resource Not Found x4, Instr NC, Context Handling, Resource Abuse
17. `396b6aa1ab86eb2e20d27582eb5eebd9` -- 6 errors: Formatting x5, Service Errors (rare)
18. `672d36d8ecc4816738433c75136eb99d` -- 6 errors: Tool Definition Issues (rare), Goal Dev, Context Handling, Language-only, Task Orch, Environment Setup

**All 21 categories are covered by these 18 traces.**

## 9. Experimentation Infrastructure

### 9.1 File Layout

```
arcgentica/
├── trail_experiment.py          # NEW: experiment runner
├── trail_prompt_v2.py           # NEW: improved prompts (versioned)
├── trail_agent.py               # MODIFIED: add single-pass mode
├── trail_common.py              # MODIFIED: experiment config
└── output/
    └── experiments/
        ├── exp_001_baseline_mini/
        │   ├── config.json
        │   ├── outputs_GAIA/          # per-trace output files
        │   └── metrics.json           # scored results
        ├── exp_002_prompt_v2/
        └── experiment_log.json        # comparative tracking
```

### 9.2 Experiment Runner (`trail_experiment.py`)

CLI entrypoint that orchestrates one experiment:

```bash
# Run on dev subset with gpt-5-mini
uv run python arcgentica/trail_experiment.py \
  --experiment-id exp_002 \
  --model openai/gpt-5-mini \
  --prompt-version v2 \
  --subset dev18 \
  --approach single_pass \
  --max-workers 5

# Run full GAIA split for final SOTA
uv run python arcgentica/trail_experiment.py \
  --experiment-id exp_final \
  --model openai/gpt-5.2 \
  --prompt-version v3 \
  --subset full \
  --approach single_pass \
  --max-workers 5
```

The runner:
1. Loads trace files (dev subset or full split)
2. Applies the selected prompt version and approach
3. Generates output files in scorer-compatible format
4. Runs `calculate_scores.py` automatically
5. Writes `metrics.json` with per-category breakdown
6. Appends to `experiment_log.json` for tracking

### 9.3 Experiment Log Format

```json
{
  "experiments": [
    {
      "experiment_id": "exp_001",
      "timestamp": "2026-02-13T22:00:00Z",
      "model": "openai/gpt-5-mini",
      "subset": "dev18",
      "prompt_version": "v1_baseline",
      "approach": "single_pass",
      "metrics": {
        "weighted_f1": 0.42,
        "location_accuracy": 0.38,
        "joint_accuracy": 0.15,
        "score_correlations": {
          "reliability": 0.65,
          "instruction_adherence": 0.30,
          "overall": 0.55
        }
      },
      "per_category_f1": {
        "Language-only": 0.55,
        "Tool-related": 0.00,
        "...": "..."
      },
      "cost_usd": 0.80,
      "wall_time_sec": 90,
      "traces_processed": 18,
      "traces_failed": 0,
      "notes": "Baseline reproduction with gpt-5-mini on dev subset"
    }
  ]
}
```

### 9.4 Dev Subset Manifest

A JSON file listing the 18 dev trace IDs so experiments are deterministic:

```json
{
  "subset_id": "dev18",
  "description": "18 representative GAIA traces covering all 21 categories",
  "trace_ids": [
    "0035f455b3ff2295167a844f04d85d34",
    "0140b3f657eddf76ca82f72c49ac8e58",
    "0adc4f3b99d9564d32811e913cc9d248",
    "18efa24e637b9423f34180d1f2041d3e",
    "27a6c5ebc3311542156fdde857a0035f",
    "3205fa0cb2135fe671bf7cd0e5a26151",
    "3637c5845140adbf29c565923c20e94d",
    "396b6aa1ab86eb2e20d27582eb5eebd9",
    "41b597524173272503073a0799ac523c",
    "59365b27641e501d105b0e8f5e7c5af7",
    "672d36d8ecc4816738433c75136eb99d",
    "6e64c2e543327a7c2f2ce3d26ced94d1",
    "72877db591837666d500b459fb3cf29d",
    "7bf0addde339e4cac9dd3b772232a7e0",
    "860f9d45f2e50bfecb190bb26eff1f32",
    "876eb108c8650d4ada63a8d39aa1e96c",
    "915d2c66879657f694f88e0ed6f02cf5",
    "99f6b447779ba86b3cff2caede832d59"
  ]
}
```

## 10. Experiment Sequence

### Phase I: Establish Prompt Ceiling (Approach A)

| Exp | Change | Expected Impact |
|-----|--------|-----------------|
| exp_001 | Reproduce baseline prompt with gpt-5-mini on dev18 | Baseline for dev subset |
| exp_002 | Add trace structure guide + location rules | +5-10% location accuracy |
| exp_003 | Add category-specific detection guidance | +10-15% F1 (unlock 0% categories) |
| exp_004 | Add 2 diverse few-shot examples | +5% joint accuracy |
| exp_005 | Replace formula scoring with LLM-generated scores | +15% score correlations |
| exp_006 | Add smart truncation for large traces | Reduce failures |
| exp_007 | Combine all improvements (best prompt) | Measure ceiling |

### Phase II: Agentic Enhancement (Approach B)

| Exp | Change | Expected Impact |
|-----|--------|-----------------|
| exp_010 | Two-pass agentic on dev18 | Compare vs best A |
| exp_011 | Tune agent budgets and chunk sizes | Optimize cost/quality |
| exp_012 | Add semantic faithfulness checks | Ensure evidence grounding |

### Phase III: Final SOTA Run

| Exp | Change | Expected Impact |
|-----|--------|-----------------|
| exp_020 | Best approach on full 117 traces with gpt-5.2 | SOTA numbers |
| exp_021 | (Optional) Re-run with anthropic/claude-opus-4-6 | A/B model comparison |

## 11. Graduation Criteria

### A to B
- Approach A's F1 plateaus for 2+ consecutive prompt iterations on dev subset
- OR >20% of dev traces fail due to context window overflow
- OR location accuracy gap vs F1 remains >15pp (indicating the model can classify but can't localize)

### Dev to Full Run
- Dev subset joint accuracy exceeds baseline by >10pp (i.e., >0.29 on dev)
- All 21 categories have >0% F1 on dev subset
- <5% trace failures

### Model Upgrade
- After approach is validated on gpt-5-mini, re-run with gpt-5.2
- Expected uplift: +5-15% across all metrics based on baseline comparison (gpt-5.2 baseline F1=0.438 vs gpt-5-mini=0.478 on subset, though different subset sizes)

## 12. Semantic Faithfulness Constraints

Per the Phase 11 execplan, all runs must also satisfy:

1. **100% valid locations**: Every predicted `location` must be a real `span_id` in the corresponding trace
2. **>=95% evidence grounding rate**: Evidence text must be findable in trace content
3. **Zero unlogged drops**: Any dropped/invalid predictions must be logged in diagnostics

These constraints are enforced by `trail_semantic_checks.py` in `strict` mode and are non-negotiable for acceptance.

## 13. Cost Estimates

| Scenario | Model | Traces | Est. Cost |
|----------|-------|--------|-----------|
| Dev iteration (1 run) | gpt-5-mini | 18 | ~$0.50-1.00 |
| Dev iteration (Approach B) | gpt-5-mini | 18 | ~$2.00-4.00 |
| Full SOTA run | gpt-5.2 | 117 | ~$15-30 |
| Full SOTA run (Approach B) | gpt-5.2 | 117 | ~$40-80 |

Budget for Phase I (7 experiments): ~$5-7
Budget for Phase II (3 experiments): ~$6-12
Budget for Phase III (1-2 experiments): ~$15-80

## 14. Success Criteria

| Metric | Baseline | Target (SOTA) |
|--------|----------|---------------|
| Weighted F1 | 0.438 | >0.55 |
| Location Accuracy | 0.478 | >0.55 |
| Joint Accuracy | 0.186 | >0.30 |
| 0% F1 Categories | 6 | 0 |
| Reliability Correlation | 0.773 | >0.80 |
| Instruction Adherence Corr. | 0.395 | >0.55 |

These targets represent meaningful improvement over the baseline while remaining achievable through prompt engineering and architectural improvements.

## 15. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OpenAI quota exhaustion during full run | Run in batches; save progress; resume from last completed trace |
| Context window overflow on large traces | Smart truncation (Section 6.2 A6); fall back to Approach B |
| Dev subset not representative of full split | Selected 18 traces covering all 21 categories with easy/medium/hard mix |
| Prompt overfitting to dev subset | Track per-category metrics; validate on held-out traces before full run |
| Agentica SDK instability | Heuristic fallback remains available; catch exceptions gracefully |
| Gold annotation inconsistencies | Handle malformed gold file (`a96c6811...`); normalize category variants |
