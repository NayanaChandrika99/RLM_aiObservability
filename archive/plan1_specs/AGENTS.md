# Specs Folder Guidelines
#
# ABOUTME: Rules for writing and maintaining specs in `specs/`.
# ABOUTME: Specs constrain the solution space; they are not implementation task lists.

## The pin: `specs/README.md`

`specs/README.md` is the spec index (“pin”) for both humans and agents.

When adding or changing a spec:

- Update `specs/README.md` in the same change.
- Keep Purpose and Keywords rich in synonyms so search hit-rate is high.
- Link to repo-relative code locations once they exist.

## Spec authoring conventions (MVP)

Each spec should be one topic per file and include, at minimum:

- **Purpose**: user-visible behavior / why it exists.
- **Non-goals**: what is explicitly excluded for MVP.
- **Interfaces**: inputs/outputs; JSON/Pydantic schemas; tool/adaptor contracts.
- **Invariants**: must-always-hold rules (correctness, safety, determinism).
- **Determinism**: seeds, IDs, timestamp normalization, replay requirements.
- **Acceptance checks**: what to run and what to observe (but do not write an ExecPlan here).
- **Keywords**: synonyms / alternate terms for retrieval.

## Plan1 MVP constraints (this repo)

Specs should keep the MVP boundaries explicit:

- Synthetic data only; scenario-driven incidents (no live RSS/web ingestion).
- In-process graph traversal (NetworkX) backed by JSON fixtures.
- Digital Twin verifier: OR-Tools (Optimization IR) + SimPy.
- LLM usage must be API-hosted only (no local model inference required).
- Side effects only through adapters; adapters support `dry_run` and emit receipts.
- Every run must produce a `RunRecord` (including early exits and failures).

## What specs are NOT

- Specs are not ExecPlans: do not embed step-by-step implementation checklists.
  Implementation plans belong in `execplan/…` and must follow `PLANS.md`.
