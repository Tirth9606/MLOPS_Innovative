"""Train a RandomForestClassifier on the student dataset and save the model.

Run from the project root:  python src/train.py
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import preprocess_data, FEATURE_COLUMNS

RANDOM_STATE = 42

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_model.pkl")


def train():
    # 1. Load + preprocess
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Train model (reproducible via random_state)
    # max_depth / min_samples_leaf regularise the forest so it generalises
    # instead of memorising the training set.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3. Quick train/test accuracy for a sanity check
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # 4. Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 5. Report
    print("Training complete.")
    print(f"  Model           : RandomForestClassifier (n_estimators=200)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")
    print(f"  Train accuracy  : {train_acc:.3f}")
    print(f"  Test accuracy   : {test_acc:.3f}")
    print(f"  Model saved to  : {os.path.relpath(MODEL_PATH, ROOT_DIR)}")

    print("\n  Feature importances:")
    for name, imp in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"    {name:<18} {imp:.3f}")

    return model


if __name__ == "__main__":
    train()
