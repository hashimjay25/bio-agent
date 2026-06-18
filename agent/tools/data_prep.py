"""
data_prep.py — Agent tool: download a CSV from the Foundry Files API,
validate columns, and engineer features ready for the ML endpoint.

The agent calls this tool first after the user uploads a CSV file.
It returns a JSON string consumed by the ml_inference tool.

Schema: cardiovascular 10-year CHD risk (Framingham-style).
"""

import io
import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd

logger = logging.getLogger(__name__)

# ── Expected feature columns (must match score.py / train.py) ────────────────
REQUIRED_COLUMNS = [
    "gender",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

# Binary categorical encodings applied client-side before sending to the endpoint.
# Must match the encodings used in azure_ml/train.py.
BINARY_ENCODINGS: dict[str, dict[str, int]] = {
    "gender": {"Female": 0, "Male": 1},
    "currentSmoker": {"No": 0, "Yes": 1},
    "BPMeds": {"No": 0, "Yes": 1},
    "prevalentStroke": {"No": 0, "Yes": 1},
    "prevalentHyp": {"No": 0, "Yes": 1},
    "diabetes": {"No": 0, "Yes": 1},
}

# Columns the dataset/model never uses — silently drop if present.
IGNORED_COLUMNS = {"TenYearCHD", "ten_year_chd", "id", "patient_id", "index"}

# Friendly column aliases the user's CSV might contain.
COLUMN_ALIASES: dict[str, str] = {
    # gender
    "Gender": "gender",
    "sex": "gender",
    "Sex": "gender",
    # age
    "Age": "age",
    # education
    "Education": "education",
    # smoking
    "current_smoker": "currentSmoker",
    "CurrentSmoker": "currentSmoker",
    "smoker": "currentSmoker",
    "Smoker": "currentSmoker",
    "cigs_per_day": "cigsPerDay",
    "CigsPerDay": "cigsPerDay",
    "cigarettesPerDay": "cigsPerDay",
    # blood pressure meds
    "bp_meds": "BPMeds",
    "BPmeds": "BPMeds",
    "bpMeds": "BPMeds",
    # stroke / hypertension
    "prevalent_stroke": "prevalentStroke",
    "PrevalentStroke": "prevalentStroke",
    "prevalent_hyp": "prevalentHyp",
    "PrevalentHyp": "prevalentHyp",
    "hypertension": "prevalentHyp",
    # diabetes
    "Diabetes": "diabetes",
    # cholesterol
    "tot_chol": "totChol",
    "TotChol": "totChol",
    "total_cholesterol": "totChol",
    "Cholesterol": "totChol",
    # blood pressure
    "SysBP": "sysBP",
    "sys_bp": "sysBP",
    "systolic_bp": "sysBP",
    "DiaBP": "diaBP",
    "dia_bp": "diaBP",
    "diastolic_bp": "diaBP",
    # BMI
    "bmi": "BMI",
    "Bmi": "BMI",
    # heart rate
    "heart_rate": "heartRate",
    "HeartRate": "heartRate",
    # glucose
    "Glucose": "glucose",
}


def _encode_binary_column(
    series: pd.Series, mapping: dict[str, int], col: str, warnings: list[str]
) -> pd.Series:
    """Encode a Yes/No or Male/Female column to 0/1, accepting already-numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        bad = ~series.dropna().isin([0, 1])
        if bad.any():
            warnings.append(
                f"Column '{col}' has numeric values outside [0, 1]; expected {sorted(mapping.values())}."
            )
        return series.astype("float")

    mapped = series.map(mapping)
    unmapped_mask = mapped.isna() & series.notna()
    if unmapped_mask.any():
        bad_values = sorted(series[unmapped_mask].astype(str).unique())[:5]
        warnings.append(
            f"Column '{col}' has unrecognised values {bad_values}; "
            f"expected one of {list(mapping.keys())}. These will be treated as missing."
        )
    return mapped.astype("float")


def _prepare_dataframe(df: pd.DataFrame) -> str:
    """Validate, encode, and normalise a cardiovascular dataframe for inference."""
    warnings: list[str] = []

    if df.empty:
        return json.dumps({"error": "The uploaded CSV is empty."})

    df = df.copy()
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Drop known non-feature columns (e.g. target, patient_id) silently.
    drop_known = [c for c in df.columns if c in IGNORED_COLUMNS]
    if drop_known:
        df.drop(columns=drop_known, inplace=True)
        warnings.append(f"Ignored non-feature columns: {drop_known}")

    # Drop any other unexpected columns.
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        df.drop(columns=extra_cols, inplace=True)
        warnings.append(f"Ignored unrecognised columns: {extra_cols}")

    # Check all required columns are present.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return json.dumps(
            {
                "error": (
                    f"CSV is missing required columns: {missing}. "
                    f"Expected columns: {REQUIRED_COLUMNS}"
                )
            }
        )

    # Encode binary categoricals (Yes/No, Male/Female) -> 0/1.
    for col, mapping in BINARY_ENCODINGS.items():
        df[col] = _encode_binary_column(df[col], mapping, col, warnings)

    # Coerce remaining numeric columns; non-parseable values become NaN.
    numeric_only_cols = [c for c in REQUIRED_COLUMNS if c not in BINARY_ENCODINGS]
    for col in numeric_only_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reorder to match model training order.
    df = df[REQUIRED_COLUMNS]

    # Drop rows where EVERY feature is missing — nothing to predict on.
    all_nan_mask = df.isna().all(axis=1)
    if all_nan_mask.any():
        n_drop = int(all_nan_mask.sum())
        df = df.loc[~all_nan_mask].reset_index(drop=True)
        warnings.append(f"Dropped {n_drop} row(s) with no usable feature values.")

    if df.empty:
        return json.dumps({"error": "No valid rows remain after cleaning."})

    # Soft range validations.
    if (df["age"].dropna() < 0).any() or (df["age"].dropna() > 120).any():
        warnings.append("Some 'age' values appear outside the expected range 0-120.")
    if (df["BMI"].dropna() < 10).any() or (df["BMI"].dropna() > 80).any():
        warnings.append("Some 'BMI' values appear outside the expected range 10-80.")
    if (df["sysBP"].dropna() < 70).any() or (df["sysBP"].dropna() > 260).any():
        warnings.append("Some 'sysBP' values appear outside the expected range 70-260 mmHg.")
    n_nan_cells = int(df.isna().sum().sum())
    if n_nan_cells:
        warnings.append(
            f"{n_nan_cells} missing cell(s) detected; the model will impute them at inference time."
        )

    # Convert NaN to None so json.dumps produces valid JSON nulls (not NaN literals).
    features = df.astype(object).where(pd.notna(df), None).values.tolist()
    preview_df = df.head(3)
    preview = preview_df.astype(object).where(pd.notna(preview_df), None).to_dict(orient="records")

    result = {
        "features": features,
        "row_count": len(features),
        "column_names": REQUIRED_COLUMNS,
        "preview": preview,
        "warnings": warnings,
    }
    logger.info(
        "CSV prep complete: %d rows, %d features, %d warnings",
        len(features),
        len(REQUIRED_COLUMNS),
        len(warnings),
    )
    return json.dumps(result)


def prep_local_csv(csv_path: str) -> str:
    """Prepare a local CSV file for inference in CLI mode."""
    try:
        df = pd.read_csv(Path(csv_path))
    except Exception as exc:
        return json.dumps({"error": f"Could not parse CSV '{csv_path}': {exc}"})

    return _prepare_dataframe(df)


# Session-scoped file roots for Foundry hosted agents.
# Files uploaded via the session /files endpoint can show up under these paths
# (preview behavior varies), or be fetched via the Foundry session-files REST API.
_SESSION_FILE_ROOTS = [Path("/files"), Path.home() / "files", Path.home(), Path("/tmp")]


def _fetch_session_csv_via_api(filename: str) -> bytes | None:
    """Download a session file via the Foundry hosted-agent REST API."""
    import os
    import requests
    from azure.identity import DefaultAzureCredential

    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").rstrip("/")
    session_id = os.environ.get("FOUNDRY_AGENT_SESSION_ID")
    agent_name = os.environ.get("FOUNDRY_AGENT_NAME", "bio-risk-agent")
    if not (endpoint and session_id):
        logger.info("_fetch_session_csv_via_api: missing endpoint or session id")
        return None

    cred = DefaultAzureCredential()
    token = cred.get_token("https://ai.azure.com/.default").token
    headers = {
        "Authorization": f"Bearer {token}",
        "Foundry-Features": "HostedAgents=V1Preview",
    }
    base = f"{endpoint}/agents/{agent_name}/endpoint/sessions/{session_id}/files"

    # If no filename given, list files and pick the first .csv.
    target = filename
    if not target:
        r = requests.get(f"{base}?api-version=v1&path=.", headers=headers, timeout=30)
        logger.info("session files list -> %s", r.status_code)
        if r.status_code >= 300:
            return None
        entries = r.json().get("entries", [])
        csvs = [e["name"] for e in entries if not e.get("is_directory") and e["name"].lower().endswith(".csv")]
        if not csvs:
            return None
        target = csvs[0]

    r = requests.get(
        f"{base}/content?api-version=v1&path={target}",
        headers={**headers, "Accept": "application/octet-stream"},
        timeout=60,
    )
    logger.info("session file download '%s' -> %s", target, r.status_code)
    if r.status_code >= 300:
        return None
    return r.content


def prep_session_csv(
    filename: Annotated[
        str,
        "Name of the CSV file the user uploaded to this session (e.g. 'biomarkers.csv'). "
        "Pass an empty string to auto-pick the first .csv in the session.",
    ] = "",
) -> str:
    """
    Prepare a CSV that the user uploaded to the current hosted-agent session.

    First tries to read the file from the session sandbox filesystem (/files, $HOME).
    Falls back to fetching it via the Foundry session-files REST API using the
    container's managed identity. Validates and engineers features ready for the
    ML endpoint.

    Returns the same JSON shape as prep_local_csv.
    """
    # 1. Try local sandbox paths first.
    candidates: list[Path] = []
    for root in _SESSION_FILE_ROOTS:
        if not root.exists():
            continue
        if filename:
            p = root / filename
            if p.is_file():
                candidates.append(p)
        else:
            candidates.extend(p for p in root.glob("*.csv") if p.is_file())

    csv_bytes: bytes | None = None
    source = ""
    if candidates:
        target = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            csv_bytes = target.read_bytes()
            source = str(target)
        except Exception as exc:
            logger.warning("failed to read %s: %s", target, exc)

    # 2. Fall back to the session-files REST API.
    if csv_bytes is None:
        csv_bytes = _fetch_session_csv_via_api(filename)
        source = f"session-api:{filename or '*.csv'}"

    if csv_bytes is None:
        return json.dumps(
            {
                "error": (
                    f"No CSV found in this session. Looked for {filename or '*.csv'} "
                    f"under {[str(r) for r in _SESSION_FILE_ROOTS]} and via the session-files API. "
                    "Make sure the file was uploaded via PUT "
                    "/agents/{name}/endpoint/sessions/{id}/files/content?path=<name>."
                )
            }
        )

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        return json.dumps({"error": f"Could not parse CSV ({source}): {exc}"})

    logger.info("prep_session_csv loaded %d bytes from %s", len(csv_bytes), source)
    return _prepare_dataframe(df)
