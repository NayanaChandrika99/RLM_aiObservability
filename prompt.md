You are auditing Phoenix-RLM Investigator as a non-production proof system.

Goal:
Determine whether the current implementation (through execplan Phase 6C) is a complete proof that recursive language-model style investigation solves long-context observability problems for:
1) Trace RCA
2) Policy-to-Trace Compliance
3) Incident Dossier

Important scope:
- Do NOT evaluate production hardening beyond current plan contract.
- Do evaluate whether the implemented system demonstrates the core long-context reasoning claim with evidence.

Read these first (in order):
1. specs/README.md
2. AGENTS.md
3. CLAUDE.md
4. execplan/phase0/master_execution_plan.md
5. ARCHITECTURE.md
6. API.md
7. DESIGN.md
8. specs/rlm_runtime_contract.md
9. specs/rlm_engines.md
10. specs/formal_contracts.md

Then inspect implementation:
- investigator/runtime/
- investigator/inspection_api/
- investigator/rca/
- investigator/compliance/
- investigator/incident/
- apps/demo_agent/
- tests/unit/
- tests/integration/

Required checks (must report each explicitly):
A) Capability proof matrix (RCA, Compliance, Incident), with columns:
- Implemented in code (yes/no + file refs)
- Contract-aligned output shape (yes/no + schema refs)
- Evidence-linked outputs (yes/no + example ref fields)
- Recursive/long-context handling mechanism (what exactly is implemented)
- Write-back implemented (yes/no + file refs)
- Tested (unit/integration/live) and what remains unproven

B) Runtime contract checks:
- RunRecord emitted on success/failure/partial
- Schema/evidence validation gates
- Recursion/budget behavior
- Sandbox behavior (actual enforcement vs signal-based simulation)

C) Data/context availability checks:
- Seeded failure dataset usability for RCA evaluation
- controls/library availability for compliance
- configs/snapshots availability for incident config-diff claims
- connectors status (explicitly staged or missing)

D) "RLM claim fidelity" check:
- Distinguish between:
  1) deterministic heuristics,
  2) recursive inspection orchestration,
  3) true LLM recursive subcall behavior.
- State clearly whether the repo currently proves "RLM fixes long-context" or only "deterministic proxy + contract scaffolding".

Execution commands:
1. uv run pytest tests/ -q -rs
2. If live checks are required and Phoenix is running:
   - PHASE4C_LIVE=1 PHOENIX_BASE_URL=http://127.0.0.1:6006 uv run pytest tests/integration/test_policy_compliance_live_writeback_phase4.py -q
   - PHASE5C_LIVE=1 PHOENIX_BASE_URL=http://127.0.0.1:6006 uv run pytest tests/integration/test_incident_dossier_live_writeback_phase5.py -q

Output format (strict):
1. Verdict: YES / PARTIAL / NO (one line)
2. What is proven today (max 10 bullets, each with file refs)
3. What is not yet proven (max 10 bullets, each with file refs)
4. Unclear or ambiguous cases resolved (explicit assumptions)
5. Minimal next 3 steps to reach "complete proof" (no production hardening; only proof gaps)
