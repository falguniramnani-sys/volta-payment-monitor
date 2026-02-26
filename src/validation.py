"""
CSV schema validation, sanitization, and fingerprinting for uploaded data.

Used by the dashboard upload flow to ensure uploaded CSVs match the expected
transaction schema before processing.
"""

import hashlib

import pandas as pd


REQUIRED_COLUMNS = [
    "id", "timestamp", "psp", "country",
    "payment_method", "card_brand", "latency_ms", "status",
]

EXPECTED_STATUSES = {"approved", "declined", "timeout"}


def validate_csv_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Validate that a DataFrame matches the expected transaction schema.

    Returns (fatal_errors, warnings) where fatal_errors block upload
    and warnings are informational only.
    """
    fatal_errors = []
    warnings = []

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fatal_errors.append(f"Missing required columns: {', '.join(missing)}")
        return fatal_errors, warnings  # Can't validate further

    # Check extra columns
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"Extra columns will be ignored: {', '.join(extra)}")

    # Check minimum row count
    if len(df) < 10:
        fatal_errors.append(f"Too few rows ({len(df)}). Minimum 10 required.")

    # Check timestamp parseability
    try:
        pd.to_datetime(df["timestamp"])
    except (ValueError, TypeError):
        fatal_errors.append("Column 'timestamp' contains unparseable values.")

    # Check latency_ms is numeric
    try:
        latency = pd.to_numeric(df["latency_ms"], errors="coerce")
        null_count = latency.isna().sum() - df["latency_ms"].isna().sum()
        if null_count > 0:
            fatal_errors.append(
                f"Column 'latency_ms' has {null_count} non-numeric values."
            )
        if (latency.dropna() < 0).any():
            fatal_errors.append("Column 'latency_ms' contains negative values.")
    except Exception:
        fatal_errors.append("Column 'latency_ms' could not be parsed as numeric.")

    # Check for unexpected status values
    unique_statuses = set(df["status"].dropna().unique())
    unexpected = unique_statuses - EXPECTED_STATUSES
    if unexpected:
        warnings.append(f"Unexpected status values: {', '.join(sorted(unexpected))}")

    # Check for empty rows
    all_null_rows = df.isnull().all(axis=1).sum()
    if all_null_rows > 0:
        warnings.append(f"{all_null_rows} empty row(s) will be dropped.")

    return fatal_errors, warnings


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a validated DataFrame:
    - Drop all-null rows
    - Parse timestamp as datetime
    - Ensure latency_ms is float
    - Keep only required columns
    """
    df = df.copy()

    # Drop all-null rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Keep only required columns
    df = df[REQUIRED_COLUMNS]

    # Parse types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").astype(float)

    return df


def compute_fingerprint(df: pd.DataFrame) -> str:
    """
    Short MD5 hash of row count + column names + first row values.
    Used as a cache key to detect when data has changed.
    """
    parts = [
        str(len(df)),
        ",".join(df.columns.tolist()),
    ]
    if len(df) > 0:
        parts.append(df.iloc[0].to_json())

    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return digest[:12]
