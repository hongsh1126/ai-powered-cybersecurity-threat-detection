"""Feature contract shared by training and inference."""

NUMERIC_FEATURES = [
    "duration_ms",
    "src_bytes",
    "dst_bytes",
    "packets",
    "failed_logins",
    "connection_rate",
    "unique_dst_ports",
]

CATEGORICAL_FEATURES = ["protocol", "service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


def missing_features(columns: list[str]) -> list[str]:
    """Return required feature names absent from a set of columns."""
    available = set(columns)
    return [name for name in FEATURES if name not in available]
