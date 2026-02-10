# ABOUTME: Exposes proof-run benchmark entrypoints for baseline-vs-RLM evaluation on frozen datasets.
# ABOUTME: Keeps reproducible proof orchestration code isolated from runtime engine modules.

from investigator.proof.benchmark import run_dataset_benchmark

__all__ = ["run_dataset_benchmark"]
