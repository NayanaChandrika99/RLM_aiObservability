# ExecPlan Folder Guidelines
#
# ABOUTME: Rules for authoring and maintaining ExecPlans in `execplan/`.
# ABOUTME: ExecPlans are executable specs a stateless agent (or novice human) can follow end-to-end.

## Source of truth

- `PLANS.md` (repo root) is the authoritative ExecPlan format and requirement spec. Follow it exactly.
- The “pin” (spec index) is `specs/README.md`.
  - ExecPlans must cite the relevant spec(s) from `specs/`.

## What belongs here

- ExecPlans only (Markdown), organized by the spec they implement.
  - Prefer: `execplan/<spec-slug>/<spec-slug>__<short-objective>.md`
  - Also acceptable (for a single plan per spec): `execplan/<spec-slug>/plan.md`
- Supporting, plan-adjacent materials that make the plan runnable are OK (small fixtures/scripts),
  but the plan file is the source of truth.

## What does NOT belong here

- Production code changes (those go in the implementation directories).
- Long-lived notes that are not actionable.
- Specs/pins/indexes (those go in `specs/`).

## Non-negotiables (per `PLANS.md`)

- Every ExecPlan is fully self-contained and beginner-readable.
- Every ExecPlan is a living document (keep `Progress`, `Surprises & Discoveries`, `Decision Log`,
  and `Outcomes & Retrospective` up to date).
- Every ExecPlan includes a `References` section with exact repo-relative file paths reviewed.
  If you cannot cite exact paths, STOP and ask Nainy.

## Loop-friendly planning (one objective per iteration)

To support “one objective per loop” execution:

- The `Progress` section must be a checkbox list where each item is:
  - atomic (doable in one iteration),
  - verifiable (includes a concrete validation command),
  - reviewable (small diff; rollbackable).
- Prefer multiple small checkboxes over one large checkbox.

## Plan1 MVP expectations (this repo)

When writing an ExecPlan for this project, keep it aligned with `plan1.md`:

- Synthetic data and scenario-driven incidents for MVP (no live RSS/web ingestion).
- Graph is file-backed (`nodes.json`/`edges.json`) and loaded into an in-memory graph (NetworkX).
- Digital Twin uses optimization (OR-Tools, schema-bound Optimization IR) + simulation (SimPy).
- LLM usage must be API-hosted only (no local model inference required).
- Side effects must go through adapters with `dry_run` and must emit receipts.

## Demo surfaces (how to choose)

Default for early milestones:

- Provide a deterministic CLI/module entrypoint that tests can call (recommended).
- Add notebooks only when needed for visualization/storytelling; keep notebooks thin and import logic
  from modules.

Promote to a web service only when you need a stable API boundary, concurrent isolation, or a UI. If a
web service is added, the ExecPlan must include:

- Endpoint contract (request/response schemas),
- a local run command,
- an integration test that asserts a stable subset of the `RunRecord`.

## Naming conventions (important)

Do not use generic phase-based naming like `phase0/`, `phase1/`, etc.

Instead, name ExecPlans after the spec(s) they implement so they remain discoverable and traceable.

Examples:

- Spec: `specs/contracts.md`
  - ExecPlans: `execplan/contracts/contracts__pydantic-models.md`, `execplan/contracts/contracts__jsonschema-export.md`
- Spec: `specs/optimization-ir.md`
  - ExecPlans: `execplan/optimization-ir/optimization-ir__ir-schema-and-validator.md`
- Spec: `specs/kg-generator.md`
  - ExecPlans: `execplan/kg-generator/kg-generator__generate-nodes-edges-twin-state.md`
