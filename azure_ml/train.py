"""
train.py — Train a cardiovascular 10-year CHD risk classifier.

Dataset: Framingham-style cardiovascular study (cardio_train.csv).
Target:  TenYearCHD  (1 = high risk of CHD within 10 years, 0 = low risk).

Features (15):
  gender              (Male/Female  -> 1/0)
  age                 (years)
  education           (ordinal 1-4)
  currentSmoker       (Yes/No       -> 1/0)
  cigsPerDay
  BPMeds              (Yes/No       -> 1/0)
  prevalentStroke     (Yes/No       -> 1/0)
  prevalentHyp        (Yes/No       -> 1/0)
  diabetes            (Yes/No       -> 1/0)
  totChol             (mg/dL)
  sysBP               (mmHg)
  diaBP               (mmHg)
  BMI
  heartRate           (bpm)
  glucose             (mg/dL)

Model: XGBoost classifier inside a sklearn Pipeline with a median imputer.
- XGBoost handles non-linear interactions well on tabular clinical data.
- scale_pos_weight handles the ~5:1 class imbalance in TenYearCHD.
- SimpleImputer(median) makes the deployed model robust to missing values
  at inference time even though XGBoost can handle NaN natively.

Run locally:
    python train.py --data ../cardio_train.csv

Run on Azure ML:
    python submit_training_job.py
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ── Reproducibility ───────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Schema ────────────────────────────────────────────────────────────────
TARGET_COL = "TenYearCHD"

# Binary categorical columns and their {category: int} encoding.
# Kept explicit so the same mapping is reproducible at inference time.
BINARY_ENCODINGS: dict[str, dict[str, int]] = {
    "gender": {"Female": 0, "Male": 1},
    "currentSmoker": {"No": 0, "Yes": 1},
    "BPMeds": {"No": 0, "Yes": 1},
    "prevalentStroke": {"No": 0, "Yes": 1},
    "prevalentHyp": {"No": 0, "Yes": 1},
    "diabetes": {"No": 0, "Yes": 1},
}

# Numeric / already-numeric columns kept as-is.
NUMERIC_COLS = [
    "age",
    "education",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

# Final ordered feature vector sent to the model.
FEATURE_COLS = [
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

CLASSES = ["low_risk", "high_risk"]


def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit Yes/No and Male/Female encodings; coerce numerics."""
    out = df.copy()
    for col, mapping in BINARY_ENCODINGS.items():
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' missing from input data.")
        out[col] = out[col].map(mapping)
    for col in NUMERIC_COLS:
        if col not in out.columns:
            raise KeyError(f"Expected numeric column '{col}' missing from input data.")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[FEATURE_COLS]


def train(data_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}   Cols: {df.shape[1]}")

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' missing from {data_path}. "
            f"Found: {list(df.columns)}"
        )

    # Drop any rows where the target is missing (these can't be used).
    n_missing_target = int(df[TARGET_COL].isna().sum())
    if n_missing_target:
        print(f"  Dropping {n_missing_target} rows with missing {TARGET_COL}.")
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X = _encode_dataframe(df)
    y = df[TARGET_COL].astype(int)

    print(f"  Class balance: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Handle imbalance: scale_pos_weight = N_negative / N_positive (on train split).
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {scale_pos_weight:.3f}  (neg={n_neg}, pos={n_pos})")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training XGBoost pipeline …")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)

    print("\nClassification Report:")
    print(report)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred): {cm}")

    # ── Feature importance (top 10) ───────────────────────────────────────
    clf: XGBClassifier = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    top = sorted(zip(FEATURE_COLS, importances), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop feature importances:")
    for name, imp in top:
        print(f"  {name:<18} {imp:.4f}")

    # ── Persist artefacts ─────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    metadata = {
        "feature_columns": FEATURE_COLS,
        "numeric_columns": NUMERIC_COLS,
        "binary_encodings": BINARY_ENCODINGS,
        "target_column": TARGET_COL,
        "classes": CLASSES,
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm,
        },
        "model": {
            "framework": "xgboost",
            "estimator": "XGBClassifier",
            "scale_pos_weight": round(scale_pos_weight, 4),
        },
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def _default_data_path() -> str:
    """Locate cardio_train.csv either next to this script or one level up."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "cardio_train.csv", here.parent / "cardio_train.csv"):
        if candidate.exists():
            return str(candidate)
    # Final fallback — let the caller see a clear FileNotFoundError on read.
    return str(here.parent / "cardio_train.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=_default_data_path(),
        help="Path to cardio_train.csv (default: workspace root or alongside train.py)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    train(data_path=args.data, output_dir=args.output_dir)
