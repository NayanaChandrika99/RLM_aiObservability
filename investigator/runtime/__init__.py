# ABOUTME: Re-exports runtime contracts and runner entrypoint for shared engine execution.
# ABOUTME: Keeps imports stable while the runtime evolves from stubs to full implementation.

from investigator.runtime.contracts import (
    ComplianceFinding,
    ComplianceReport,
    DatasetRef,
    EvidenceRef,
    IncidentDossier,
    IncidentHypothesis,
    IncidentTimelineEvent,
    InputRef,
    RCAReport,
    RecommendedAction,
    RepresentativeTrace,
    RunRecord,
    RuntimeBudget,
    SuspectedChange,
)
from investigator.runtime.runner import run_engine

__all__ = [
    "ComplianceFinding",
    "ComplianceReport",
    "DatasetRef",
    "EvidenceRef",
    "IncidentDossier",
    "IncidentHypothesis",
    "IncidentTimelineEvent",
    "InputRef",
    "RCAReport",
    "RecommendedAction",
    "RepresentativeTrace",
    "RunRecord",
    "RuntimeBudget",
    "SuspectedChange",
    "run_engine",
]
