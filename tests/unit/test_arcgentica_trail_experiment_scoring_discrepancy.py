# ABOUTME: Verifies TRAIL experiment scoring behavior when gold JSON contains minor formatting defects.
# ABOUTME: Ensures scorer keeps file accounting consistent by tolerating trailing-commas in gold files.

from __future__ import annotations

import json
from pathlib import Path

from arcgentica.trail_experiment import _score_outputs


def test_score_outputs_accepts_trailing_comma_gold_json(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace_a.json"
    trace_file.write_text("{}", encoding="utf-8")

    gold_dir = tmp_path / "gold"
    gen_dir = tmp_path / "gen"
    gold_dir.mkdir()
    gen_dir.mkdir()

    gold_with_trailing_comma = """{
  "trace_id": "trace_a",
  "errors": [
    {
      "category": "Formatting Errors",
      "location": "span_1",
      "evidence": "format mismatch",
      "description": "bad format",
      "impact": "LOW"
    },
  ],
  "scores": [
    {
      "reliability_score": 2,
      "security_score": 5,
      "instruction_adherence_score": 2,
      "plan_opt_score": 2,
      "overall": 2.75
    }
  ]
}
"""
    (gold_dir / "trace_a.json").write_text(gold_with_trailing_comma, encoding="utf-8")

    generated = {
        "trace_id": "trace_a",
        "errors": [
            {
                "category": "Formatting Errors",
                "location": "span_1",
                "evidence": "format mismatch",
                "description": "bad format",
                "impact": "LOW",
            }
        ],
        "scores": [
            {
                "reliability_score": 2,
                "security_score": 5,
                "instruction_adherence_score": 2,
                "plan_opt_score": 2,
                "overall": 2.75,
            }
        ],
    }
    (gen_dir / "trace_a.json").write_text(json.dumps(generated), encoding="utf-8")

    metrics = _score_outputs(gold_dir=gold_dir, generated_dir=gen_dir, trace_files=[trace_file])

    assert metrics["files_processed"] == 1
    assert metrics["joint_accuracy"] == 1.0
