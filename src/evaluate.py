"""Evaluate the trained model on the test set and save a report.

Computes accuracy, precision, recall, F1-score, confusion matrix and a full
classification report. Results are printed and written to reports/.
Run from the project root:  python src/evaluate.py
"""

import os
import json
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data_preprocessing import preprocess_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_model.pkl")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# "PASS" is treated as the positive class for precision/recall/F1
POSITIVE_LABEL = "PASS"


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Train it first with:  python src/train.py"
        )

    model = joblib.load(MODEL_PATH)

    # Use the same reproducible split as training -> X_test / y_test
    _, X_test, _, y_test = preprocess_data()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)

    labels = ["FAIL", "PASS"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_text = classification_report(y_test, y_pred, labels=labels)

    # --- Print ---
    print("Evaluation results (positive class = PASS):")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"           pred_FAIL  pred_PASS")
    print(f"  FAIL     {cm[0][0]:>9}  {cm[0][1]:>9}")
    print(f"  PASS     {cm[1][0]:>9}  {cm[1][1]:>9}")
    print("\nClassification report:")
    print(report_text)

    # --- Save reports (JSON for metrics, TXT for human-readable report) ---
    os.makedirs(REPORT_DIR, exist_ok=True)

    metrics = {
        "accuracy": round(acc, 4),
        "precision_pass": round(prec, 4),
        "recall_pass": round(rec, 4),
        "f1_pass": round(f1, 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(REPORT_DIR, "evaluation_report.txt"), "w") as f:
        f.write("Student Performance Model - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy : {acc:.3f}\n")
        f.write(f"Precision: {prec:.3f}\n")
        f.write(f"Recall   : {rec:.3f}\n")
        f.write(f"F1-score : {f1:.3f}\n\n")
        f.write("Confusion matrix (rows=actual, cols=predicted):\n")
        f.write(f"           pred_FAIL  pred_PASS\n")
        f.write(f"  FAIL     {cm[0][0]:>9}  {cm[0][1]:>9}\n")
        f.write(f"  PASS     {cm[1][0]:>9}  {cm[1][1]:>9}\n\n")
        f.write("Classification report:\n")
        f.write(report_text + "\n")

    print(f"Saved metrics  -> {os.path.relpath(os.path.join(REPORT_DIR, 'metrics.json'), ROOT_DIR)}")
    print(f"Saved report   -> {os.path.relpath(os.path.join(REPORT_DIR, 'evaluation_report.txt'), ROOT_DIR)}")


if __name__ == "__main__":
    evaluate()
