"""
ml_inference.py — Agent tool: send prepared feature rows to the Azure ML
Managed Online Endpoint and return predictions with summary statistics.

Rows are sent in batches (≤100 per request) to stay within the endpoint's
HTTP payload limit.
"""

import json
import logging
import math
import os
from typing import Annotated

import requests
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # max rows per Azure ML request
DEFAULT_AAD_SCOPE = "https://ml.azure.com/.default"


def run_bio_inference(
    features_json: Annotated[
        str,
        (
            "JSON string produced by prep_session_csv or prep_local_csv. "
            "Must contain 'features' (list of lists) and 'row_count' keys."
        ),
    ],
) -> str:
    """
    Call the Azure ML bio-risk endpoint with prepared feature rows.

    Returns a JSON string with:
      - predictions:   list of int  (0=low_risk, 1=high_risk) per row
      - labels:        list of str  ("low_risk" or "high_risk") per row
      - probabilities: list of [P(low), P(high)] per row
      - row_count:     total rows scored
      - summary:       aggregate stats (high_risk_count, low_risk_count, high_risk_pct)
      - top_high_risk_rows: indices of the 5 rows with the highest high-risk probability
    """
    # ── Load endpoint config from environment ─────────────────────────────
    endpoint_url = os.environ.get("ML_ENDPOINT_URL", "")
    auth_mode = os.environ.get("ML_AUTH_MODE", "aad").strip().lower()
    aad_scope = os.environ.get("ML_AAD_SCOPE", DEFAULT_AAD_SCOPE)

    if not endpoint_url:
        return json.dumps(
            {
                "error": (
                    "ML_ENDPOINT_URL must be set. "
                    "Run azure_ml/deploy.py and copy the value into your .env file."
                )
            }
        )

    if auth_mode != "aad":
        return json.dumps(
            {
                "error": (
                    "Unsupported ML_AUTH_MODE. Set ML_AUTH_MODE=aad for Azure AD authentication."
                )
            }
        )

    # ── Parse input ────────────────────────────────────────────────────────
    try:
        prep_result = json.loads(features_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid features_json — not valid JSON: {exc}"})

    if "error" in prep_result:
        return json.dumps({"error": f"Data prep error: {prep_result['error']}"})

    features: list[list[float]] = prep_result.get("features", [])
    if not features:
        return json.dumps({"error": "No feature rows found in features_json."})

    # ── Batch inference ────────────────────────────────────────────────────
    all_predictions: list[int] = []
    all_probs: list[list[float]] = []
    all_labels: list[str] = []

    n_batches = math.ceil(len(features) / BATCH_SIZE)
    try:
        credential = DefaultAzureCredential()
        aad_token = credential.get_token(aad_scope).token
    except Exception as exc:
        return json.dumps(
            {
                "error": (
                    "Failed to obtain Azure AD token for Azure ML endpoint. "
                    "Ensure managed identity or az login is available and has endpoint invoke permissions. "
                    f"Details: {exc}"
                )
            }
        )

    headers = {
        "Authorization": f"Bearer {aad_token}",
        "Content-Type": "application/json",
    }

    for batch_idx in range(n_batches):
        batch = features[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        payload = json.dumps({"data": batch})

        try:
            response = requests.post(
                endpoint_url,
                data=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return json.dumps(
                {"error": f"Azure ML endpoint timed out on batch {batch_idx + 1}/{n_batches}."}
            )
        except requests.exceptions.RequestException as exc:
            return json.dumps(
                {
                    "error": (
                        f"Azure ML endpoint request failed on batch {batch_idx + 1}/{n_batches}: "
                        f"{exc}"
                    )
                }
            )

        try:
            result = response.json()
        except ValueError:
            return json.dumps(
                {"error": f"Azure ML returned non-JSON on batch {batch_idx + 1}: {response.text[:200]}"}
            )

        # Some endpoint stacks return a JSON-encoded string payload.
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return json.dumps(
                    {
                        "error": (
                            f"Azure ML returned a string payload that is not valid JSON on batch {batch_idx + 1}. "
                            f"Payload preview: {result[:200]}"
                        )
                    }
                )

        if not isinstance(result, dict):
            return json.dumps(
                {
                    "error": (
                        f"Azure ML returned unsupported payload type {type(result).__name__} "
                        f"on batch {batch_idx + 1}."
                    )
                }
            )

        if "error" in result:
            return json.dumps({"error": f"Endpoint error: {result['error']}"})

        try:
            all_predictions.extend(result["predictions"])
            all_probs.extend(result["probabilities"])
            all_labels.extend(result["labels"])
        except KeyError as exc:
            return json.dumps(
                {
                    "error": (
                        f"Azure ML response missing expected key {exc} on batch {batch_idx + 1}. "
                        f"Response keys: {list(result.keys())}"
                    )
                }
            )

    # ── Build summary statistics ───────────────────────────────────────────
    high_risk_count = sum(p == 1 for p in all_predictions)
    low_risk_count = len(all_predictions) - high_risk_count
    high_risk_pct = round(100 * high_risk_count / len(all_predictions), 1)

    # Top-5 rows with highest high-risk probability
    high_risk_probs = [row[1] for row in all_probs]  # P(high_risk) is index 1
    top_indices = sorted(
        range(len(high_risk_probs)),
        key=lambda i: high_risk_probs[i],
        reverse=True,
    )[:5]

    summary = {
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "high_risk_pct": high_risk_pct,
    }

    logger.info(
        "Inference complete: %d rows — %d high-risk (%.1f%%)",
        len(all_predictions),
        high_risk_count,
        high_risk_pct,
    )

    return json.dumps(
        {
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probs,
            "row_count": len(all_predictions),
            "summary": summary,
            "top_high_risk_rows": top_indices,
        }
    )


def score_session_csv(
    filename: Annotated[
        str,
        (
            "Name of the CSV file the user uploaded to this session "
            "(e.g. 'biomarkers.csv'). Pass an empty string to auto-pick the "
            "most recent .csv in the session sandbox."
        ),
    ] = "",
) -> str:
    """
    One-shot tool: prepare the user's uploaded session CSV AND score it against
    the Azure ML bio-risk endpoint. Use this whenever the user asks for risk
    predictions, top-risk indexes, or analysis of a CSV they have uploaded.

    Internally chains prep_session_csv + run_bio_inference — the agent does NOT
    need to (and must not) move the feature payload between tools manually.

    Returns a JSON string with:
      - row_count
      - column_names
      - warnings
      - predictions, labels, probabilities
      - summary { high_risk_count, low_risk_count, high_risk_pct }
      - top_high_risk_rows (indices of the 5 highest high-risk probabilities)
    """
    from tools.data_prep import prep_session_csv

    prep_json = prep_session_csv(filename=filename)
    prep_obj = json.loads(prep_json)
    if "error" in prep_obj:
        return json.dumps({"error": prep_obj["error"]})

    score_json = run_bio_inference(features_json=prep_json)
    score_obj = json.loads(score_json)
    if "error" in score_obj:
        return json.dumps(
            {
                "error": score_obj["error"],
                "row_count": prep_obj.get("row_count"),
                "warnings": prep_obj.get("warnings", []),
            }
        )

    # Merge prep metadata + inference output into one response.
    merged = {
        "row_count": prep_obj.get("row_count"),
        "column_names": prep_obj.get("column_names"),
        "warnings": prep_obj.get("warnings", []),
        **score_obj,
    }
    return json.dumps(merged)
