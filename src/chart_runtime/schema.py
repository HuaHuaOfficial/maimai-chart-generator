"""Draft IR storage specification, not a parser or legality implementation.

Arrays are device-resident. Stable IDs survive permitted revisions; table row
indices may change. Variable-count notes, Tracks and segments use offsets.
Time uses one nanosecond rounding policy pinned by the schema definition.
"""

SCHEMA_ID = "chart-ir/1-draft"

TABLES = {
    "events": (
        ("event_id", "int64"), ("time_ns", "int64"),
        ("note_begin", "int64"), ("note_end", "int64"),
    ),
    "notes": (
        ("note_id", "int64"), ("event_id", "int64"),
        ("kind", "uint8"), ("sensor_id", "int16"),
        ("modifiers", "uint16"), ("hold_duration_ns", "int64"),
    ),
    "tracks": (
        ("track_id", "int64"), ("event_id", "int64"),
        # -1 means a headless Track; its geometry is still complete.
        ("head_note_id", "int64"), ("start_sensor_id", "int16"),
        ("wait_ns", "int64"), ("segment_begin", "int64"), ("segment_end", "int64"),
    ),
    "segments": (
        ("segment_id", "int64"), ("track_id", "int64"),
        ("route_id", "int32"), ("duration_ns", "int64"),
    ),
    "tempo": (("time_ns", "int64"), ("bpm_num", "int64"), ("bpm_den", "int64")),
}

# These are transport requirements, not presently implemented feature/rule IDs.
WITNESS_FIELDS = (
    "rule_id", "severity", "observed_value", "limit_value", "excess",
    "object_ids", "dependency_interval_ns", "scope_complete", "coverage_id",
)
