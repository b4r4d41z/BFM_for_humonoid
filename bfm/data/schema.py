"""BFM schema v2 (stored using the Zarr v2 storage format)."""
from bfm.contracts.kuavo import STATE_FIELDS, KuavoContract

SCHEMA_NAME = "bfm.measured.transitions"
SCHEMA_VERSION = "2.0.0"


def episode_metadata(contract: KuavoContract, *, episode_id, split, provenance, initial_scene):
    return {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id, "split": split, "provenance": provenance,
        "state_source": "measured_simulator", "coordinate_frame": "environment_local",
        "contract": contract.to_dict(), "initial_scene": initial_scene,
        "state_fields": {k: {"width": w, "unit": u} for k, (w, u) in STATE_FIELDS.items()},
        "action_units": ["rad"] * 7 + ["binary_closed_signed"],
        "terminal_semantics": "terminated_no_bootstrap_truncated_bootstrap_pre_reset",
    }
