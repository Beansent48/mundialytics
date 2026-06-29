"""Data quality auditing utilities for Mundialytics.

The data-quality layer is offline-only and does not change model behavior.
"""

from .data_audit import (
    DATA_AUDIT_VERSION,
    DataAuditOutputs,
    audit_data_sources,
    write_data_audit_outputs,
)
from .entity_guardrails import (
    ENTITY_GUARDRAILS_VERSION,
    EntitySquadGuardrailOutputs,
    build_entity_squad_guardrails,
)
from .match_dataset_foundation import (
    MATCH_DATASET_FOUNDATION_VERSION,
    MatchDatasetFoundationOutputs,
    prepare_match_dataset,
)
from .model_ready_snapshots import (
    MODEL_READY_SNAPSHOTS_VERSION,
    ModelReadySnapshotsOutputs,
    build_model_ready_match_snapshots,
)

__all__ = [
    "DATA_AUDIT_VERSION",
    "DataAuditOutputs",
    "audit_data_sources",
    "write_data_audit_outputs",
    "ENTITY_GUARDRAILS_VERSION",
    "EntitySquadGuardrailOutputs",
    "build_entity_squad_guardrails",
    "MATCH_DATASET_FOUNDATION_VERSION",
    "MatchDatasetFoundationOutputs",
    "prepare_match_dataset",
    "MODEL_READY_SNAPSHOTS_VERSION",
    "ModelReadySnapshotsOutputs",
    "build_model_ready_match_snapshots",
]

