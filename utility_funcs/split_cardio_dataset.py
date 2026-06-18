"""
split_cardio_dataset.py — Split the cardiovascular dataset into train/inference sets.

- 80% → cardio_train.csv       (includes the TenYearCHD target, feeds train.py)
- 20% → cardio_inference.csv   (TenYearCHD removed, used for inference)

Stratified on the target to preserve the ~85/15 class balance in both splits.

Reads from <repo>/data/ and writes the splits back to <repo>/data/, regardless
of the working directory the script is invoked from.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "cardio vascular_dataset.csv"
TRAIN_OUT = DATA_DIR / "cardio_train.csv"
INFER_OUT = DATA_DIR / "cardio_inference.csv"
TARGET = "TenYearCHD"
RANDOM_STATE = 42
TEST_SIZE = 0.20

df = pd.read_csv(SRC)

# Drop rows where the target itself is missing — they can't be used for
# supervised training or for scoring inference predictions.
missing_target = df[TARGET].isna().sum()
if missing_target:
    print(f"Dropping {missing_target} rows with missing {TARGET}.")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

train_df, infer_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=df[TARGET],
)

train_df.to_csv(TRAIN_OUT, index=False)

infer_features = infer_df.drop(columns=[TARGET])
infer_features.to_csv(INFER_OUT, index=False)

print(f"Source rows:      {len(df)}")
print(f"Training rows:    {len(train_df)}  -> {TRAIN_OUT.relative_to(DATA_DIR.parent)} (includes {TARGET})")
print(f"Inference rows:   {len(infer_df)}  -> {INFER_OUT.relative_to(DATA_DIR.parent)} (no {TARGET})")
print()
print("Class balance (train):")
print(train_df[TARGET].value_counts(normalize=True).round(4).to_string())
print()
print("Class balance (inference held-out):")
print(infer_df[TARGET].value_counts(normalize=True).round(4).to_string())
