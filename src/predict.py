"""Predict PASS/FAIL for a single student using the trained model.

Usage (from the project root):

  # Use the built-in example student:
  python src/predict.py

  # Or pass the 7 features in order:
  # study_hours attendance previous_marks assignment_score internal_score sleep_hours participation
  python src/predict.py 6 87 72 81 76 7 8
"""

import os
import sys
import joblib
import pandas as pd

from data_preprocessing import FEATURE_COLUMNS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_model.pkl")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Train it first with:  python src/train.py"
        )
    return joblib.load(path)


def predict_student(features: dict, model=None):
    """Predict for one student given a dict of the 7 features.

    Returns (label, probability_of_label).
    """
    if model is None:
        model = load_model()

    # Order the features exactly as the model expects
    row = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    label = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    # Probability corresponding to the predicted label
    prob_of_label = proba[list(model.classes_).index(label)]
    return label, prob_of_label


def main():
    args = sys.argv[1:]

    if len(args) == len(FEATURE_COLUMNS):
        values = [float(a) for a in args]
        student = dict(zip(FEATURE_COLUMNS, values))
    elif len(args) == 0:
        # Built-in example (matches the Phase 1 brief)
        student = {
            "study_hours": 6,
            "attendance": 87,
            "previous_marks": 72,
            "assignment_score": 81,
            "internal_score": 76,
            "sleep_hours": 7,
            "participation": 8,
        }
    else:
        print(f"Error: expected {len(FEATURE_COLUMNS)} values in this order:")
        print("  " + " ".join(FEATURE_COLUMNS))
        sys.exit(1)

    model = load_model()
    label, prob = predict_student(student, model)

    print("Input student:")
    for k, v in student.items():
        print(f"  {k:<18} {v}")
    print()
    print(f"Prediction: {label}")
    print(f"Probability: {prob * 100:.0f}%")


if __name__ == "__main__":
    main()
