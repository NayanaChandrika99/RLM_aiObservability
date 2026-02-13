ABOUTME: Slide-by-slide deck script for presenting an RLM-based AI observability investigation layer at PepsiCo.
ABOUTME: Includes on-slide bullets, visualization descriptions, and consolidated speaker notes.

# AI Observability Investigation Layer + RLM (PepsiCo) Deck Script (v2)

## Assumptions (edit to match PepsiCo reality)
- PepsiCo has an internal AI observability platform for LLM/agent systems.
- The system presented here is an **AI Observability Investigation Layer** that runs RCA and policy compliance over observability evidence.
- Phoenix (Arize Phoenix) is the trace store / UI used for agent traces in this repo.
- The goal is to present a credible "working beta" story for RCA + Policy Compliance, with a clear enterprise rollout path.

## Design System (Evidence Ledger Style)
Goal: avoid generic "AI network" templates and make the deck feel like a serious, auditable investigation system.

- Visual metaphor: an "evidence ledger" that records bounded investigations over traces.
- Tone: executive-clean + engineering-credible (more evidence, less decoration).
- Design principle: every claim should have a place to anchor (evidence chips, run-record stamp, or a metric tile).

**Canvas and layout**
- 16:9 widescreen.
- Safe margins: 48px all sides.
- Grid: 60/40 split on most slides (left narrative, right evidence visual).
- One primary visual per slide; no background hero gradients.
- Card system: 12px radius, 1px stroke, subtle shadow (one style reused everywhere).
- Spacing rhythm: 8px micro, 16px standard, 24px section, 40px hero gaps (keep it consistent).

**Color and contrast**
- Background: warm paper `#FBF7F0` with faint ledger lines `#1B2A3A` at 5-8% opacity (or `#D7CBBE` at ~18% if you need stronger projector contrast).
- Primary text: deep ink navy `#0B1F33`.
- Secondary text: slate `#54657A`.
- Borders/grid: `#D7CBBE` at 60-80% opacity.
- Accent (attention only): signal red `#C62828` used only for risk/fail/delta/attention.
- Link/highlight accent (optional): teal `#1B7F79` used for clickable evidence highlights.
- Status colors (optional): `succeeded #2E7D32`, `partial #C47F17`, `failed #C62828`.

**Typography**
- Titles: modern sans, heavy weight (Aptos Display / Segoe UI Semibold).
- Body: modern sans (Aptos / Segoe UI).
- Evidence and IDs: monospace (Cascadia Mono / Consolas), used for `trace_id`, `span_id`, tool names, budgets.
- Typographic trick: treat evidence chips like inline citations (monospace, compact, high-contrast) to make the deck feel provable.

**Reusable components**
- Ledger header: small top-left label (e.g., "AI Observability Investigation Layer") with a thin vertical rule.
- Evidence chips: rounded pills with monospace text (e.g., `trace_id: ...`, `span_id: ...`, `artifact: ...`).
- Budget strip: a compact bar showing remaining `depth / iterations / tool_calls / cost`.
- RunRecord stamp: a small "RUN RECORDED" stamp badge to reinforce auditability.
- Trace mini-map: small node-link or tree outline with highlighted spans (only as a watermark or small card).
- Table style: thin grid lines, subtle zebra rows, one accent column for deltas.
- "Stack frame" cards: nested rounded rectangles with objective + budget strip + typed return.
- "Run health" tiles: `succeeded / partial / failed` shown as labeled counters (not charts).
- Delta bars: thin horizontal bars with a tick at zero; red only if negative.
- Tool allowlist strip: chips with tool names and one-line purpose (keeps the runtime concrete).

**Do / don't**
- Do: make the investigation feel like a program (stack frames, return values, typed JSON).
- Do: keep diagrams clean (single stroke weight, consistent arrows, no 3D).
- Don't: use stock "AI network" backgrounds, heavy gradients, or big diagonal stripes.
- Don't: use uncleared brand/product names in the visuals; keep platform naming generic.

## Slide 1 - Title
**Title:** AI Observability Investigation Layer (RLM-powered)

**Subtitle:** From long-context bottlenecks to low-cost automated RCA and policy compliance

**Visual**
- Layout:
  - Left 60%: title + subtitle + a single-line promise statement.
  - Right 40%: faint trace mini-map watermark (5% opacity), clipped to a rounded rectangle.
- Background:

  - No gradient bands.
- Accent:  - Warm paper with ledger lines.
  - One thin vertical red rule at the far left of the title block (8px wide).
- Micro-diagram (keep small, not a footer):
  - Under subtitle: three evidence chips connected by a thin line: `Traces` -> `RLM Runtime` -> `Findings`.
  - The chips should look like "labels" on a ledger, not buttons.
- Footer:
  - Bottom-right: `Slide 1/12` in light slate + a small tag: `evidence`.
- Brand handling:
  - If brand-safe: small "PepsiCo" label in the top-left, not a logo hero.
  - If not brand-safe: omit branding entirely and keep it generic.

---

## Slide 2 - AI Observability: The Long-Context Problem
**On-slide bullets**
- AI incidents are not a single log line; evidence is distributed across:
  - trace trees (parent/child spans)
  - tool I/O artifacts and errors
  - retrieval chunks and citations
  - configs/snapshots and diffs
  - policy controls and required evidence lists
- "Long context" fails in practice:
  - the important clue is often buried ("lost in the middle")
  - stuffing everything into one prompt increases cost + confusion
  - repeated manual triage does not scale at PepsiCo volume
- We need automation that is evidence-grounded, budgeted, and replayable.

**Visual**
- Layout:
  - Left 55-60%: bullets with one inline "lost in the middle" callout pill.
  - Right 40-45%: a tall "Evidence Ledger" card that reads like an internal ops worksheet.
- Evidence ledger card (right):
  - Header row: `Signal` | `What it contains` | `Volume` | `Where the clue hides`.
  - 8-10 rows: `trace spans`, `tool I/O`, `retrieval chunks`, `config diffs`, `controls`, `messages`, `timeouts`, `schema errors`, `rate limits`, `upstream deps`.
  - Middle row highlight:
    - Put the "critical clue" row around the middle (e.g., `retrieval chunks`).
    - Add a red bracket on the right margin plus label: "lost in the middle".
    - Add a faint "scroll fade" at the top/bottom of the card to imply overflow.
- Context-stuffing inset (bottom-right of the visual area):
  - Two tiny cards side-by-side:
    - `One-shot prompt`: cramped textbox + warning triangle + `tokens ↑, clarity ↓`.
    - `Bounded investigation`: three ledger rows labeled `plan -> inspect -> submit` + a tiny budget strip.
- Micro-detail to sell credibility:
  - Add 2-3 evidence chips under the ledger card (monospace): `trace_id`, `span_id`, `artifact_id` (placeholders are fine).
- Build order:
  1. Ledger card appears with rows.
  2. Red bracket animates onto the middle row.
  3. Inset cards appear (one-shot vs bounded).

---

## Slide 3 - What An RLM Is (Practical Definition)
**On-slide bullets**
- An RLM is not just a model call; it's **model + runtime**:
  - plan next step
  - gather evidence via tools
  - optionally delegate sub-investigations
  - synthesize, then finalize into a schema
- Key property: **recursive decomposition** instead of monolithic prompting.
- Guardrails are part of the design:
  - budgets (iterations, depth, tools, tokens, cost, wall-time)
  - sandbox / tool allowlist
  - schema validation + run artifacts

**Visual**
- Layout:
  - Left: bullets.
  - Right: "runtime as a ledger" (stacked steps + budget + typed return).
- Runtime ledger (right):
  - Five step cards stacked vertically with row numbers `01`..`05` in a left gutter:
    - `PLAN` (context snapshot)
    - `TOOL_CALL` (allowlisted inspection)
    - `DELEGATE_SUBCALL` (fresh stack frame)
    - `SYNTHESIZE` (merge evidence into draft)
    - `FINALIZE` (schema-validated submit)
  - Each step card shows one tiny field line in monospace to feel program-like, e.g.:
    - `objective: "label trace failure mode"`
    - `tool: list_spans(trace_id=...)`
    - `subcall: hypothesis="retrieval_failure"`
    - `patch: add evidence_refs[2]`
    - `submit: RCAReport`
- Budget strip (under the stack):
  - Compact strip: `depth 1/2 | iter 2/8 | tools 3/16 | cost $0.01 | wall 12s`.
  - Add a tiny "remaining" bar (battery-style) for either cost or wall-time.
- Typed return (bottom of the right column):
  - A "return value" card with a 2-3 line monospace JSON snippet and a green `SCHEMA OK` stamp.
  - Add a subtle `RUN RECORDED` stamp beside it (reinforces artifacts).
- Build order:
  1. Step cards appear top-to-bottom.
  2. Budget strip fades in.
  3. Return value card stamps `SCHEMA OK`.

---

## Slide 4 - RLM vs Agentic Tool-Use: What's Actually Different?
**On-slide bullets**
- **Most tool agents** behave like a growing chat transcript (linear loop, messy history).
- **This RLM runtime** behaves like a program with isolation + contracts:
  1. Scope isolation: child calls run in fresh contexts (like stack frames), return structured "return values".
  2. Tree control-flow: explicit recursion depth + subcall budgets prevent rabbit holes.
  3. Typed actions + typed outputs: actions are validated (tool_call / delegate_subcall / synthesize / finalize); outputs must match JSON contracts.
  4. Deterministic narrowing before LLM: reduce noise and cost; LLM operates on a focused evidence set.
  5. Auditability: every run writes a run record (trajectory, subcalls, usage, evidence pointers).

**Visual**
- Layout: 50/50 split visual, with an optional one-line "so what" under the title.
- Left column: "Typical tool agent (linear transcript)"
  - One tall chat transcript card with:
    - alternating chat bubbles and tool output blocks,
    - two "retry" markers (crossed-out),
    - a fake scrollbar to imply endless growth,
    - a red micro-warning badge near the bottom: `context drift risk`.
  - One small metric chip under it: `context length: 18k tokens` (illustrative).
- Right column: "RLM runtime (program-like)"
  - Three nested stack-frame cards (like function call frames):
    - Frame 0: objective + budget strip
    - Frame 1: delegated subcall + its own budget strip
    - Frame 2: faint (optional) to show depth limit
  - Each frame contains:
    - a one-line tool-call record (monospace),
    - a return-value chip: `return: { evidence_refs: [...] }`.
  - Add a small pool-budget chip near the bottom: `shared budget: cost <= $X` (illustrative).
- Bottom strip: "Contracts + audit trail"
  - Three compact chips: `schema validated`, `run_record.json`, `subcall metadata`.
  - Keep the slide mostly monochrome; use green only for `schema validated`.
- Build order:
  1. Left transcript fills downward (messy).
  2. Right stack frames appear (structured).
  3. Contracts/audit strip appears last.

---

  ## Slide 5 - Capability 1: Root Cause Analysis (RCA)
  **On-slide bullets**
  - RCA goal: assign primary failure mode with evidence pointers and remediation.
  - Why it matters:
    - reduces MTTR (mean time to resolution)
    - standardizes incident classification across teams
    - makes investigations reproducible and measurable
  - Output (structured, evidence-linked):
    - `primary_label` (tool failure, retrieval failure, schema mismatch, upstream dependency, instruction failure)
    - `evidence_refs` (`trace_id`, `span_id`, artifact refs)
    - summary + remediation + gaps

  **Visual**
  - Layout:
    - Left: bullets.
    - Right: RCA output + "why this label" evidence panel.
  - RCA output card (right, top):
    - Header row: `RCAReport` (monospace) + `RUN RECORDED` stamp.
    - Main body:
      - `primary_label` as a bold rectangular tag (not a pill).
      - `confidence` as a thin meter with 3 tick marks (low/med/high).
      - Evidence chips section:
        - 1 chip for `trace_id`
        - 2 chips for `span_id` (the specific hot spans)
        - 1 chip for `artifact_id` (tool I/O or retrieval)
      - `remediation` section: 2 short bullets.
      - `gaps` section: 1 line in slate (keeps honesty).
  - Evidence panel (right, bottom):
    - Mini "Hot spans" ranking table:
      - columns: `rank`, `span`, `signal`, `why it matters`
      - highlight top 2 rows that match the evidence chips
    - Trace mini-map to the right of the table:
      - simple tree outline with highlighted nodes; labels only on highlighted nodes
      - use one accent (teal) for highlights; use red only if span status is ERROR
  - Micro-callout on the left:
    - Add one small callout box under bullets: "Click-through evidence in the observability UI (span + artifact)."
  - Build order:
    1. RCAReport appears.
    2. Evidence chips appear one-by-one.
    3. Hot spans table highlights the matching rows.

---

## Slide 6 - Capability 2: Policy-to-Trace Compliance
**On-slide bullets**
- Compliance goal: evaluate behavior against explicit controls using trace evidence.
- Why it matters:
  - governance: controls need provable evidence, not just narrative
  - audit: structured decisions with evidence pointers
  - standardization: consistent evaluation across apps/teams
- Output (structured, per-control + overall):
  - verdict: `pass/fail/needs_review/insufficient_evidence`
  - confidence + severity + missing evidence list
  - evidence refs mapped to spans/artifacts

**Visual**
- Layout:
  - Left: bullets.
  - Right: compliance control ledger + "missing evidence" + "policy pack" hint.
- Control ledger table (right, top):
  - Header row pinned: `Overall verdict: PASS` (or `NEEDS REVIEW`) with a small confidence meter.
  - Table columns: `control_id` | `verdict` | `confidence` | `evidence refs`.
  - 4-6 representative controls as rows (don’t overcrowd):
    - `verdict` as a small rectangular badge (green/amber/red).
    - `evidence refs` as 2-3 chips per row: `span_id`, `artifact_id`.
  - Add a subtle rightmost column with a checkbox icon for "evidence complete" (checked vs empty).
- Side drilldown (right, bottom-left):
  - Card titled `Missing evidence`:
    - 2-3 items with empty checkboxes (e.g., `tool_output artifact`, `retrieval citations`, `policy attestation`).
    - one line: `missing -> insufficient_evidence` (explains deterministic precedence).
- Policy pack hint (right, bottom-right):
  - Small "Policy pack" card that looks like a document tab:
    - label: `controls/v1` (generic)
    - one line: `scoped by app + tools`
- Build order:
  1. Overall verdict header appears.
  2. Table rows populate top-to-bottom.
  3. Missing evidence card appears (when relevant) to show explainability.

---

## Slide 7 - How RLM Tackles Long Context for RCA + Compliance
**On-slide bullets**
- Step 1: deterministic narrowing
  - hot spans (errors, exceptions, latency) / scoped controls
  - reduces noise and prevents context poisoning
- Step 2: evidence-driven recursive loop
  - fetch only needed evidence via tools
  - use semantic subcalls for focused synthesis (not everything)
  - delegate subcalls for subproblems (hypotheses / missing evidence)
- Step 3: finalize into contracts
  - schema-validated JSON output
  - run record emitted for replay and audit

**Visual**
- Layout:
  - Left: bullets.
  - Right: funnel-to-tree visual that explicitly shows "reduce" then "recursively inspect".
- Top of the visual: raw evidence pile
  - A scattered cloud of faint evidence chips: `span`, `tool_io`, `chunk`, `config`, `control`.
  - Label: `raw evidence (too big for one prompt)`.
- Stage 1 card: `Deterministic narrowing`
  - A funnel mouth that collapses into a small stack of highlighted chips:
    - `8 hot spans`
    - `6 scoped controls`
  - Count annotation in monospace: `120 spans -> 8 hot`.
- Stage 2 card: `Evidence-driven recursion`
  - A compact recursion tree:
    - Parent node: `objective`
    - Child nodes: `tool_call`, `delegate_subcall`
    - Leaf node: `submit`
  - Add a tiny budget strip next to the tree: `depth <= 2 | iter <= 8`.
- Stage 3 card: `Typed finalize`
  - A submit card with a schema-check stamp and 2-3 evidence chips attached like citations.
- Bounded cost callout:
  - Small badge: `bounded cost/run` with a `cost <= $X` chip (illustrative).
- Build order:
  1. Evidence cloud appears.
  2. Funnel collapses into highlighted chips.
  3. Recursion tree draws in.
  4. Submit + schema stamp appears.

---

## Slide 8 - Implementation: What The Investigation Layer Is Doing Under the Hood
**On-slide bullets**
- Investigator runtime is a bounded REPL loop:
  - `call_tool(...)` = read-only inspection API fetch (allowlisted)
  - `llm_query(...)` = short semantic synthesis calls
  - `SUBMIT(...)` = finalize structured output
- Safety:
  - sandbox tool allowlist and argument validation
  - budgets (iterations, depth, tools, tokens, cost, wall-time)
  - deterministic recovery submit near deadline (avoid partial/hangs)
- Auditability:
  - every run writes run artifacts: `run_record.json` + output JSON
  - trajectory, usage/cost, evidence pointers, errors

**Visual**
- Layout:
  - Left: bullets.
  - Right: architecture + a REPL panel that makes the runtime feel real (not abstract).
- Architecture row (right, top):
  - Four ledger cards with clean arrows:
    1. `Trace store + artifacts` (label generic: "observability UI / export job")
    2. `Investigation queue` (async trigger)
    3. `Investigator worker` (capability = RCA / compliance)
    4. `Write-back findings` (annotations + evidence links)
  - Keep arrow style consistent: 1px stroke, rounded arrowheads, no gradients.
- Runtime internals (right, middle):
  - One big `RLM REPL runtime` card with four sub-badges:
    - `ToolRegistry` (allowlisted)
    - `Sandbox` (argument checks)
    - `Budget` (depth/iter/cost/wall)
    - `RunRecord` (artifacts)
  - Optional: a small chip inside the budget badge: `recovery submit near deadline`.
- REPL panel (right, bottom-left):
  - Mock "code cell" card with 4-6 lines (monospace), e.g.:
    - `spans = call_tool("list_spans", trace_id=...)`
    - `...`
    - `SUBMIT({...})`
  - Add a timer chip in the corner: `wall-time 18s`.
- Tool allowlist strip (right, bottom-right):
  - 6-8 chips with tool names + 1-line purpose beneath (tiny slate text), e.g.:
    - `list_spans` (trace graph)
    - `get_span` (details)
    - `get_tool_io` (tool evidence)
    - `list_controls` (policy pack)
    - `required_evidence` (control needs)
- Auditability footer strip (under the whole right visual):
  - Three compact chips: `state_trajectory`, `usage/cost`, `subcall_metadata`.
- Build order:
  1. Architecture row appears.
  2. Runtime card expands with internal badges.
  3. REPL panel + allowlist strip appear last.

---

## Slide 9 - Results: Proof Canaries (Quality + Run Health + Cost)
**On-slide bullets (use these exact numbers from artifacts)**
- RCA 5-trace canary:
  - baseline accuracy `0.0` -> RLM `0.8` (delta `+0.8`)
  - run health: `succeeded 5/5`, `wall-time partials 0`
  - cost: total `~$0.0216` (avg `~$0.0043/trace`)
- Compliance 5-trace canary:
  - baseline accuracy `0.8` -> RLM `1.0` (delta `+0.2`)
  - run health: `succeeded 5/5`, `partial_rate 0.0`, `failed 0`
  - cost: total `~$0.0469` (avg `~$0.0094/trace`)
- Key takeaway: working beta with strong run-health and measurable quality lift.

**Visual**
- Layout:
  - Left: bullets with exact metrics.
  - Right: a single scorecard that looks like an internal benchmark report page.
- Main scorecard (right):
  - Table rows: `RCA` and `Policy compliance`.
  - Columns:
    - `baseline accuracy`
    - `RLM accuracy`
    - `delta` (thin delta bar + zero tick)
    - `run health` (mini tiles: `succeeded`, `partial`, `failed`)
    - `avg cost/trace` (monospace)
  - Make `RLM accuracy` and `delta` visually dominant:
    - larger type
    - heavier weight
    - delta bar right-aligned for easy scanning
- Run-health micro-visual:
  - Use compact counters per row: `S 5`, `P 0`, `F 0` (monospace, color-coded).
- Evidence callout (bottom-left of the right area):
  - "Evidence" header + two file chips with the internal paths.
  - Add a tiny `RUN RECORDED` stamp next to the chips.
- Build order:
  1. Table appears without deltas.
  2. Delta bars animate in.
  3. Evidence chips appear last.

**Internal evidence paths**
- RCA: `artifacts/proof_runs/phase10-rca-only-canary-5-retrieverfix-20260211T021018Z/rca_only_report.json`
- Compliance: `artifacts/proof_runs/phase10-compliance-only-canary-5-20260211T023031Z/compliance_only_report.json`

---

## Slide 10 - Scaling to Company-Wide PepsiCo Observability
**On-slide bullets**
- Deployment model:
  - async job queue triggered by traces/incidents
  - investigator workers per capability (RCA, compliance; incident dossier later)
  - write-back findings to the AI observability platform UI with evidence links
- Guarded rollout:
  - policy packs per domain/team
  - budgets per run + per team + per day
  - route low-confidence/high-severity to humans
  - continuous benchmarking gates (quality + run-health + cost)
- Economics (pilot profile estimate):
  - combined avg `~$0.0137/trace` (RCA + compliance)
  - scale by increasing automation of repetitive, high-volume patterns.

**Visual**
- Layout:
  - Left: bullets.
  - Right: operating model diagram + rollout timeline (clean, not crowded).
- Two-lane operating model (right, top):
  - Lane A header: `Automation (bounded)`:
    - `Trace trigger -> Queue -> Worker pool -> Findings -> Write-back`
    - Put a small budget envelope badge at the start: `daily cap`, `per-run budget`, `wall-time`.
  - Lane B header: `Human review (only when needed)`:
    - `low confidence / high severity -> analyst -> approval/escalation`
    - Add a small "evidence pack" icon (a stack of chips) to show what humans receive.
- Governance elements (embedded, not a separate slide):
  - `Policy packs` card feeding into the worker pool (looks like a folder tab).
  - `Benchmark gates` chip: `quality + run health + cost`.
- Rollout timeline (right, bottom):
  - Three ledger milestones with success metrics under each:
    1. `Canary` (N small) + `non-negative delta`
    2. `Guarded rollout` (team opt-in) + `budget caps`
    3. `Standard workflow` (org-wide) + `SLA/MTTR impact`
- Build order:
  1. Automation lane appears.
  2. Human lane appears (to show "humans stay in the loop").
  3. Timeline appears last.

---

## Slide 11 - Roadmap: What Comes Next (Credible, Not Hype)
**On-slide bullets**
- Improve RCA edge cases (e.g., tool timeout vs upstream dependency classification)
- Enforce stricter RCA REPL submit contract (prevent missing `primary_label`)
- Add connectors for logs/metrics (beyond traces) when available
- Expand incident dossier engine once multi-trace correlation tools are ready

**Visual**
- Layout:
  - Left: bullets.
  - Right: Now/Next/Later ledger with explicit success metrics (no hype words).
- Now/Next/Later (right):
  - Three vertical columns; each column is a ledger card with 2-3 item cards inside.
  - Each item card contains:
    - title (short, specific)
    - why (one line, slate)
    - metric (one line, monospace)
    - evidence chip placeholder (e.g., `proof_report.json`)
  - Keep the items phrased like tickets, e.g.:
    - `disambiguate tool timeout vs upstream`
    - `enforce submit contract`
    - `add logs/metrics connectors`
    - `expand incident correlation tools`
- Bottom rule banner:
  - A tiny banner across the bottom: `Rule: improvements must show non-negative delta on proof canaries`.
- Build order:
  1. Now column appears.
  2. Next column appears.
  3. Later column appears.
  4. Bottom rule banner fades in.

---

## Slide 12 - References (External + Internal)
**External (put in appendix / speaker notes)**
- RLM paper: `https://arxiv.org/abs/2512.24601`
- Lost in the Middle: `https://arxiv.org/abs/2307.03172`
- OpenTelemetry signals: `https://opentelemetry.io/docs/concepts/signals/`
- NIST AI RMF: `https://www.nist.gov/itl/ai-risk-management-framework`
- Phoenix docs: `https://arize.com/docs/phoenix`
- RLM explainer: `https://alexzhang13.github.io/blog/2025/rlm/`

**Internal evidence**
- RCA canary report: `artifacts/proof_runs/phase10-rca-only-canary-5-retrieverfix-20260211T021018Z/rca_only_report.json`
- Compliance canary report: `artifacts/proof_runs/phase10-compliance-only-canary-5-20260211T023031Z/compliance_only_report.json`

**Visual**
- Layout:
  - Left column: external references as citation cards (no big raw URLs).
  - Right column: internal evidence as proof chips (file paths) plus a run-record stamp motif.
- External citations (left):
  - 4-6 small cards with:
    - paper/spec title (bold),
    - one line: why it matters,
    - a small tag: `arxiv` / `spec` / `concept`.
  - Keep it readable for execs: titles + why; URLs can live in speaker notes/appendix.
- Internal proof index (right):
  - One "Proof index" card with:
    - 2 file chips (RCA + compliance) and a one-line note: `replayable from artifacts`.
    - a small `RUN RECORDED` stamp.
  - Optional: QR placeholder for an internal wiki page (only if you have one).

---

## Speaker Notes (All Slides)

### Slide 1
- "This is a recursive, evidence-driven investigation layer on top of our AI observability stack. It turns messy trace evidence into structured, auditable RCA and compliance findings."

### Slide 2
- Emphasize that "more tokens" is not the solution: it increases spend and still misses structured investigation steps.
- The enterprise pain is "repetitive investigations" and "inconsistent decisions."

### Slide 3
- "Think of it like a program that calls functions and returns structured values, not a chat bot generating text."

### Slide 4
- Be honest: "An RLM is still an agentic system, but it is a more constrained, auditable, hierarchical form of agentic execution."
- The real differentiator is engineering: isolation, budgets, typed interfaces, and replay artifacts.

### Slide 5
- Emphasize "evidence linked to trace spans" so engineers can click through in observability UI.

### Slide 6
- Compliance isn't "trust the model." It's "prove the model's decision with trace evidence."

### Slide 7
- This is the core "long context automation" idea: search the evidence space instead of stuffing tokens.

### Slide 8
- "This is built for operations: you can inspect the execution trajectory and costs for each run."

### Slide 9
- If asked "why baseline is low": baseline is a simple deterministic heuristic; we use it only to measure non-regression and delta.

### Slide 10
- Emphasize "this becomes an observability investigation layer" not just a one-off agent.

### Slide 11
- Make clear the difference between "working runtime" and "perfect accuracy." We have a measurable path to improve.
