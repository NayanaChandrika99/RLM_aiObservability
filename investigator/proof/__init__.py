# ABOUTME: Exposes proof-run benchmark entrypoints for baseline-vs-RLM evaluation on frozen datasets.
# ABOUTME: Keeps reproducible proof orchestration code isolated from runtime engine modules.

from investigator.proof.benchmark import run_dataset_benchmark
from investigator.proof.repl_canary import run_phase10_repl_canary_proof

__all__ = ["run_dataset_benchmark", "run_phase10_repl_canary_proof"]
