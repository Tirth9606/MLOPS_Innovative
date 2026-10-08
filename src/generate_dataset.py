"""Generate a realistic synthetic student performance dataset.

The target (PASS/FAIL) is driven by a weighted combination of the features
plus random noise, so the relationships are realistic but NOT deterministic.
Run from the project root:  python src/generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

# Reproducibility
RANDOM_STATE = 42
N_RECORDS = 1000

# Resolve paths relative to the project root (one level up from src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "student_data.csv")


def generate(n=N_RECORDS, seed=RANDOM_STATE):
    """Create a DataFrame of synthetic students with a PASS/FAIL target."""
    rng = np.random.default_rng(seed)

    # --- Features with realistic ranges ---
    study_hours = np.clip(rng.normal(5, 2, n), 0, 12)           # hours/day
    attendance = np.clip(rng.normal(78, 15, n), 30, 100)        # percent
    previous_marks = np.clip(rng.normal(65, 15, n), 10, 100)    # percent
    assignment_score = np.clip(rng.normal(70, 15, n), 0, 100)   # percent
    internal_score = np.clip(rng.normal(68, 15, n), 0, 100)     # percent
    sleep_hours = np.clip(rng.normal(7, 1.3, n), 3, 11)         # hours/night
    participation = np.clip(rng.normal(6, 2, n), 0, 10)         # 0-10 scale

    # --- Weighted "ability" score (normalised feature contributions) ---
    # Higher study hours, attendance, marks, scores, participation help.
    # Sleep helps up to ~7-8h (modelled as closeness to an 7.5h optimum).
    sleep_quality = 1 - np.abs(sleep_hours - 7.5) / 4.5

    score = (
        0.22 * (study_hours / 12)
        + 0.20 * (attendance / 100)
        + 0.20 * (previous_marks / 100)
        + 0.15 * (assignment_score / 100)
        + 0.13 * (internal_score / 100)
        + 0.05 * sleep_quality
        + 0.05 * (participation / 10)
    )

    # Add realistic noise so the target is not perfectly separable
    noise = rng.normal(0, 0.025, n)
    final_score = score + noise

    # Threshold chosen to give a reasonable PASS/FAIL balance (~60/40)
    performance = np.where(final_score >= 0.62, "PASS", "FAIL")

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_marks": previous_marks.round(1),
        "assignment_score": assignment_score.round(1),
        "internal_score": internal_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "participation": participation.round(1),
        "performance": performance,
    })
    return df


def main():
    df = generate()
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)

    print(f"Generated {len(df)} records -> {os.path.relpath(RAW_PATH, ROOT_DIR)}")
    print("\nClass balance:")
    print(df["performance"].value_counts())
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
