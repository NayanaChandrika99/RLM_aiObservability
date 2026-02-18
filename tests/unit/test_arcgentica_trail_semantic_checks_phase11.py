# ABOUTME: Verifies semantic faithfulness enforcement for TRAIL predictions in Phase 11.
# ABOUTME: Covers strict drop-or-repair behavior for invalid span references and ungrounded evidence.

from __future__ import annotations

from arcgentica.trail_semantic_checks import enforce_semantic_faithfulness


def test_strict_mode_repairs_location_and_evidence() -> None:
    trace_payload = {
        "trace_id": "trace_a",
        "spans": [
            {
                "span_id": "root_span",
                "span_name": "main",
                "status_code": "Unset",
                "span_attributes": {"detail": "request timed out waiting for dependency"},
                "logs": [],
                "child_spans": [],
            }
        ],
    }
    prediction = {
        "trace_id": "trace_a",
        "errors": [
            {
                "category": "Timeout Issues",
                "location": "missing_span",
                "evidence": "this text is not in trace",
                "description": "timeout happened",
                "impact": "HIGH",
            }
        ],
        "scores": [{"overall": 3.0}],
    }

    repaired, report = enforce_semantic_faithfulness(
        trace_payload=trace_payload,
        prediction=prediction,
        mode="strict",
    )

    assert len(repaired["errors"]) == 1
    assert repaired["errors"][0]["location"] == "root_span"
    assert "timed out" in repaired["errors"][0]["evidence"].lower()
    assert report["repair_actions"]["location_repaired"] == 1
    assert report["repair_actions"]["evidence_repaired"] == 1
    assert report["dropped_errors"] == 0


def test_strict_mode_drops_error_when_location_cannot_be_repaired() -> None:
    trace_payload = {"trace_id": "trace_b", "spans": []}
    prediction = {
        "trace_id": "trace_b",
        "errors": [
            {
                "category": "Timeout Issues",
                "location": "unknown_span",
                "evidence": "timed out",
                "description": "timeout happened",
                "impact": "HIGH",
            }
        ],
        "scores": [{"overall": 3.0}],
    }

    repaired, report = enforce_semantic_faithfulness(
        trace_payload=trace_payload,
        prediction=prediction,
        mode="strict",
    )

    assert repaired["errors"] == []
    assert report["dropped_errors"] == 1
    assert report["drop_reasons"]["unrepairable_location"] == 1


def test_off_mode_leaves_prediction_unchanged() -> None:
    trace_payload = {"trace_id": "trace_c", "spans": []}
    prediction = {
        "trace_id": "trace_c",
        "errors": [
            {
                "category": "Timeout Issues",
                "location": "unknown_span",
                "evidence": "timed out",
                "description": "timeout happened",
                "impact": "HIGH",
            }
        ],
        "scores": [{"overall": 3.0}],
    }

    unchanged, report = enforce_semantic_faithfulness(
        trace_payload=trace_payload,
        prediction=prediction,
        mode="off",
    )

    assert unchanged == prediction
    assert report["mode"] == "off"
