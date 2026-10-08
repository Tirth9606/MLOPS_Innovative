"""Train a RandomForestClassifier on the student dataset and save the model.

Phase 2: training runs are tracked with MLflow (parameters, metrics and the
trained model artifact) using a LOCAL file-based tracking store (./mlruns).
No remote/cloud server is required.

Run from the project root:  python src/train.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_preprocessing import preprocess_data, FEATURE_COLUMNS

RANDOM_STATE = 42

# Model hyperparameters (kept as variables so they can be logged to MLflow)
N_ESTIMATORS = 200
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 8

# "PASS" is the positive class for precision / recall / F1
POSITIVE_LABEL = "PASS"

# MLflow settings (all local)
EXPERIMENT_NAME = "student-performance-prediction"
REGISTERED_MODEL_NAME = "student-performance-model"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_model.pkl")

# Local MLflow backend. MLflow 3.x put the old file store in "maintenance mode",
# so we use a LOCAL SQLite database instead: it is still fully local (no cloud /
# remote server) and, unlike the file store, it supports the Model Registry so
# model versioning works. Artifacts are stored in a local ./mlartifacts folder.
# Paths are resolved relative to the project root so the commands work from any cwd.
TRACKING_URI = "sqlite:///" + os.path.join(ROOT_DIR, "mlflow.db").replace(os.sep, "/")
ARTIFACT_URI = "file:///" + os.path.join(ROOT_DIR, "mlartifacts").replace(os.sep, "/")


def train():
    # Point MLflow at the LOCAL SQLite backend and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    # Create the experiment with a local artifact location if it does not exist yet
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load + preprocess
    X_train, X_test, y_train, y_test = preprocess_data()

    with mlflow.start_run() as run:
        # 2. Train model (reproducible via random_state)
        # max_depth / min_samples_leaf regularise the forest so it generalises
        # instead of memorising the training set.
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 3. Metrics on the held-out test set
        y_pred = model.predict(X_test)
        train_acc = model.score(X_train, y_train)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        rec = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)

        # 4. Log parameters + metrics to MLflow
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
        })
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        })

        # 5. Log the trained model as an MLflow artifact. We try to register it
        #    in the Model Registry for versioning; if the local store does not
        #    support the registry we fall back to logging without registration.
        input_example = X_train.head(2)
        # cloudpickle keeps the classic sklearn serialisation (MLflow 3.x otherwise
        # defaults to skops, which rejects RandomForest's internal tree types).
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",                       # MLflow 3.x API (not deprecated artifact_path)
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            registered = True
        except MlflowException:
            # Registry unsupported by this backend -> log model without registering
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            registered = False

        # 6. Also save a plain .pkl so Phase 1 predict.py / evaluate.py still work
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # 7. Report
        print("Training complete.")
        print(f"  Model            : RandomForestClassifier (n_estimators={N_ESTIMATORS})")
        print(f"  Training samples : {len(X_train)}")
        print(f"  Testing samples  : {len(X_test)}")
        print(f"  Train accuracy   : {train_acc:.3f}")
        print(f"  Test accuracy    : {acc:.3f}")
        print(f"  Precision (PASS) : {prec:.3f}")
        print(f"  Recall (PASS)    : {rec:.3f}")
        print(f"  F1-score (PASS)  : {f1:.3f}")
        print(f"  Model saved to   : {os.path.relpath(MODEL_PATH, ROOT_DIR)}")

        print("\n  MLflow tracking:")
        print(f"    Experiment     : {EXPERIMENT_NAME}")
        print(f"    Run ID         : {run.info.run_id}")
        print(f"    Tracking store : {TRACKING_URI} (local SQLite)")
        if registered:
            print(f"    Registered as  : '{REGISTERED_MODEL_NAME}' (new version created)")
        else:
            print(f"    Model logged as a run artifact (registry not supported by local store)")

        print("\n  Feature importances:")
        for name, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ):
            print(f"    {name:<18} {imp:.3f}")

    return model


if __name__ == "__main__":
    train()
