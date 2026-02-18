ABOUTME: Canonical working notes for TRAIL split/protocol and benchmark behavior used in this repo.
ABOUTME: Keeps paper details and repo-level realities together to prevent split/evaluation mistakes.

# TRAIL Working Notes (Canonical for this Workspace)

## Source of truth used here

- Paper: `trail-benchmark/TRAIL.pdf`
- Benchmark runner: `trail-benchmark/benchmarking/run_eval.py`
- Benchmark scorer: `trail-benchmark/benchmarking/calculate_scores.py`
- Data folders:
  - Raw traces: `trail-benchmark/benchmarking/data/GAIA/`, `trail-benchmark/benchmarking/data/SWE Bench/`
  - Gold labels: `trail-benchmark/benchmarking/processed_annotations_gaia/`, `trail-benchmark/benchmarking/processed_annotations_swe_bench/`

## What TRAIL means by "split"

TRAIL uses **dataset-source split**, not train/val/test:

- `GAIA`
- `SWE Bench`

In code, `--split` in `run_eval.py` is only this selector (`GAIA` or `SWE Bench`) and runs over the full folder for that source split.

## Exact dataset composition (paper + repo reality)

### Paper claims

- TRAIL benchmark contains **148 traces** and **841 annotated errors**.
- Drawn from GAIA and SWE-Bench-Lite.
- Section 4.3 says errors found in **114 GAIA traces** and **30 SWE Bench traces**.

### Local repository counts (current clone)

- Raw trace files:
  - `data/GAIA`: **117** files
  - `data/SWE Bench`: **31** files
  - Total: **148**
- Gold annotation files:
  - `processed_annotations_gaia`: **117** files
  - `processed_annotations_swe_bench`: **31** files

### Important paper inconsistency to track

- `Table 5` text extraction shows `GAIA Total Traces = 118` and `SWEBench Total Traces = 31` (sum=149), which conflicts with:
  - abstract/main text total of 148, and
  - actual repo files (117 + 31 = 148).

For implementation/evaluation in this workspace, treat repo data folders as canonical.

## Evaluation protocol used by TRAIL code

1. `run_eval.py` generates one output JSON per raw trace for the selected split.
2. `calculate_scores.py` compares those outputs against the corresponding `processed_annotations_*` gold directory.
3. Reported metrics include:
   - Weighted category F1 (multi-label)
   - Average location accuracy
   - Average location-category joint accuracy
   - Pearson correlations on rubric scores

No official train/dev/test partition is defined in the released benchmark code.

## Scoring behavior details that matter for experiments

- Category labels are normalized using case/space and loose substring matching (`normalize_category`).
- Location and joint accuracy are per-trace ratios over gold sets.
- "No-error" gold traces contribute 0 to location/joint due to scorer logic.
- Scorer extracts JSON by regex from model text (`extract_json_from_text`), so extra non-JSON text can still parse if a JSON object exists.

## Data quality quirks in this repo copy

- Malformed gold JSON file:
  - `processed_annotations_gaia/a96c6811716c0473b86a23321db79c34.json`
  - Has a trailing comma in `errors` list and fails strict JSON parse.
- Some category strings in gold labels are misspelled/variant forms and only partially normalized by scorer.

## Practical implication for our "strict split" work

- If we want strict train/val/test discipline, it must be an **internal split we create** on GAIA trace IDs.
- Final reported TRAIL result should still be run against the full TRAIL GAIA split to stay comparable to published numbers.

## Useful commands

Count files:

```bash
ls trail-benchmark/benchmarking/data/GAIA/*.json | wc -l
ls trail-benchmark/benchmarking/data/'SWE Bench'/*.json | wc -l
```

Run official eval for GAIA:

```bash
cd trail-benchmark/benchmarking
python3 run_eval.py --model=<litellm-model-id> --data_dir=data --output_dir=results --split=GAIA
python3 calculate_scores.py --results_dir=results
```
