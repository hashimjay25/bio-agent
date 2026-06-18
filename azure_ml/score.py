"""
score.py — Azure ML Managed Online Endpoint scoring script.

init()  : called once at container startup to load the model.
run()   : called per inference request.

Model: cardiovascular 10-year CHD risk classifier (XGBoost).
Class semantics: 0 = low_risk (no CHD within 10 years), 1 = high_risk (CHD).

Request schema:
    {"data": [[gender, age, education, currentSmoker, cigsPerDay, BPMeds,
               prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
               diaBP, BMI, heartRate, glucose], ...]}

    Categorical encoding (must be applied client-side):
      gender:           Female=0, Male=1
      currentSmoker:    No=0, Yes=1
      BPMeds:           No=0, Yes=1
      prevalentStroke:  No=0, Yes=1
      prevalentHyp:     No=0, Yes=1
      diabetes:         No=0, Yes=1
    Missing values are accepted (sent as null/NaN); the model imputes them.

Response schema:
    {
      "predictions":   [0, 1, ...],           // 0=low_risk, 1=high_risk
      "probabilities": [[0.82, 0.18], ...],   // [P(low_risk), P(high_risk)]
      "labels":        ["low_risk", "high_risk", ...]
    }
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL = None
CLASSES = ["low_risk", "high_risk"]

FEATURE_COLUMNS = [
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


def init():
    """Load model artifacts once at container start."""
    global MODEL
    # Azure ML injects the registered model path via AZUREML_MODEL_DIR
    model_root = Path(os.getenv("AZUREML_MODEL_DIR", "outputs"))
    direct_path = model_root / "model.pkl"

    logger.info("AZUREML_MODEL_DIR: %s", model_root)

    # Registered models are often mounted as: <root>/<model_name>/<version>/<artifact_name>
    candidate_paths = [direct_path]
    if model_root.exists():
        candidate_paths.extend(sorted(model_root.rglob("model.pkl")))

        # Fallback: support custom model artifact names like bio-risk-model.pkl
        pkl_candidates = sorted(model_root.rglob("*.pkl"))
        candidate_paths.extend([p for p in pkl_candidates if p not in candidate_paths])

    for path in candidate_paths:
        if path.exists():
            MODEL = joblib.load(path)
            logger.info("Model loaded successfully from %s", path)
            return

    available_files = []
    if model_root.exists():
        available_files = [str(p) for p in model_root.rglob("*") if p.is_file()]

    raise FileNotFoundError(
        "Could not find model.pkl under AZUREML_MODEL_DIR. "
        f"Checked root: {model_root}. Found files: {available_files[:20]}"
    )


def run(raw_data: str) -> str:
    """
    Perform batch inference.

    Args:
        raw_data: JSON string with schema {"data": [[feature_row], ...]}

    Returns:
        JSON string with predictions, probabilities, and class labels.
    """
    try:
        if MODEL is None:
            return json.dumps({"error": "Model not initialized."})

        payload = json.loads(raw_data)
        data = payload["data"]

        if not isinstance(data, list) or len(data) == 0:
            return json.dumps({"error": "Field 'data' must be a non-empty list of feature rows."})

        X = np.array(data, dtype=float)

        # Guard: each row must have exactly the right number of features
        if X.shape[1] != len(FEATURE_COLUMNS):
            return json.dumps(
                {
                    "error": (
                        f"Each row must have {len(FEATURE_COLUMNS)} features "
                        f"in order: {FEATURE_COLUMNS}"
                    )
                }
            )

        # Wrap in DataFrame so the sklearn pipeline (fitted with feature names)
        # gets the columns it expects and skips the missing-feature-names warning.
        X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        predictions = MODEL.predict(X_df).tolist()
        probabilities = MODEL.predict_proba(X_df).tolist()
        labels = [CLASSES[p] for p in predictions]

        return json.dumps(
            {
                "predictions": predictions,
                "probabilities": probabilities,
                "labels": labels,
            }
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.exception("Inference error")
        return json.dumps({"error": str(exc)})
