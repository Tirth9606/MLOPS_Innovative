"""FastAPI service that serves the trained student-performance model.

Phase 3: exposes the Phase 1/2 model over a REST API.

Endpoints:
  GET  /health   -> API + model status (and model version if available)
  POST /predict  -> PASS/FAIL prediction + probability for one student

The prediction reuses the SAME helpers as the CLI (src/predict.py), so the
feature ordering applied at serving time matches training exactly — there is
no training/serving preprocessing mismatch.

Run locally from the project root:
  uvicorn api.app:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Make the Phase 1 code importable, regardless of current working dir ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import load_model, predict_student, FEATURE_COLUMNS  # noqa: E402

REGISTERED_MODEL_NAME = "student-performance-model"

# --- Load the model once at startup (reused for every request) ---
try:
    MODEL = load_model()
    MODEL_LOADED = True
    LOAD_ERROR = None
except Exception as exc:  # model file missing / unreadable
    MODEL = None
    MODEL_LOADED = False
    LOAD_ERROR = str(exc)


def get_model_version():
    """Best-effort: latest registered version from the local MLflow registry.

    Returns a version string, or None if MLflow / the registry is unavailable
    (e.g. inside a minimal Docker image). The API works fine either way.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        db_path = os.path.join(ROOT_DIR, "mlflow.db")
        if not os.path.exists(db_path):
            return None
        mlflow.set_tracking_uri("sqlite:///" + db_path.replace(os.sep, "/"))
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if not versions:
            return None
        latest = max(int(v.version) for v in versions)
        return str(latest)
    except Exception:
        return None


# --- Request / response schemas (Pydantic validation) ---
class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=24, description="Average study hours per day")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    previous_marks: float = Field(..., ge=0, le=100, description="Previous exam marks (%)")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment score (%)")
    internal_score: float = Field(..., ge=0, le=100, description="Internal assessment score (%)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep per night (hours)")
    participation: float = Field(..., ge=0, le=10, description="Class participation (0-10)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "study_hours": 6,
                "attendance": 87,
                "previous_marks": 72,
                "assignment_score": 81,
                "internal_score": 76,
                "sleep_hours": 7,
                "participation": 8,
            }
        }
    }


class PredictionOutput(BaseModel):
    prediction: str
    probability: float


app = FastAPI(
    title="Student Performance Prediction API",
    description="Serves the trained Random Forest model (PASS/FAIL).",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Report API status, whether the model is loaded, and its version."""
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_version": get_model_version() if MODEL_LOADED else None,
        "detail": LOAD_ERROR,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    """Predict PASS/FAIL for one student and return the probability."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first with: python src/train.py",
        )

    # Pydantic guarantees all 7 fields are present and in range
    features = student.model_dump()
    label, prob = predict_student(features, MODEL)

    return PredictionOutput(prediction=str(label), probability=round(float(prob), 2))
