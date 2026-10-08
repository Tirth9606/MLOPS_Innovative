"""Load, validate and preprocess the student dataset.

Exposes helper functions used by train.py / evaluate.py:
  - load_data()          : read the raw CSV (with validation + missing handling)
  - preprocess_data()    : split into train/test feature sets
Run directly for a quick sanity check:  python src/data_preprocessing.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paths relative to the project root (one level up from src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "student_data.csv")

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_score",
    "sleep_hours",
    "participation",
]
TARGET_COLUMN = "performance"


def load_data(path=RAW_PATH):
    """Read the dataset, validate columns and handle missing values."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Generate it first with:  python src/generate_dataset.py"
        )

    df = pd.read_csv(path)

    # Validate that all expected columns are present
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Handle missing values: numeric features filled with column median,
    # rows with a missing target are dropped (cannot learn from them).
    df = df.dropna(subset=[TARGET_COLUMN])
    for col in FEATURE_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_data(path=RAW_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Return clean X_train, X_test, y_train, y_test.

    RandomForest does not require feature scaling, so we keep the features
    in their natural units and only perform the train/test split. The split
    is stratified on the target to preserve the PASS/FAIL ratio.
    """
    df = load_data(path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Preprocessing complete.")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")
    print(f"  Features        : {list(X_train.columns)}")
    print(f"  Train class balance:\n{y_train.value_counts().to_string()}")
