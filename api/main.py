"""
Smart Agriculture Assistant API (Phase 6).

Ties the four trained models (crop recommendation, fertilizer
recommendation, irrigation prediction, disease detection) together
behind one HTTP API, plus endpoints to preview the IoT simulator
(Phase 5) and run its combined field-report demo.

Run with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.

Each model is loaded once at startup, not per-request. If a model's
training artifacts aren't present (e.g. you've only run some of the
train.py scripts), that model's endpoints return a 503 rather than
crashing the whole API — the other endpoints keep working.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import config
from api.schemas import (
    CropRecommendationRequest,
    FertilizerRecommendationRequest,
    IrrigationPredictionRequest,
    FieldReportRequest,
)
from src.crop_recommendation.predict import CropRecommender
from src.fertilizer_recommendation.predict import FertilizerRecommender
from src.irrigation_prediction.predict import IrrigationPredictor
from src.disease_detection.predict import DiseaseDetector
from src.iot.simulator import IoTSensorSimulator
from src.iot.adapters import (
    to_crop_recommendation_input,
    to_fertilizer_recommendation_input,
    to_irrigation_prediction_input,
)

# Holds whichever models loaded successfully at startup. A model that
# isn't trained yet is simply absent here (None), not a startup crash.
models: dict = {
    "crop": None,
    "fertilizer": None,
    "irrigation": None,
    "disease": None,
}


def _try_load(name: str, loader):
    try:
        models[name] = loader()
        print(f"[startup] Loaded {name} model.")
    except FileNotFoundError as e:
        print(f"[startup] {name} model not available yet: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load("crop", CropRecommender)
    _try_load("fertilizer", FertilizerRecommender)
    _try_load("irrigation", IrrigationPredictor)
    _try_load("disease", DiseaseDetector)
    yield
    models.clear()


app = FastAPI(
    title="Smart Agriculture Assistant API",
    description=(
        "Crop recommendation, fertilizer recommendation, irrigation "
        "prediction, and multi-crop disease detection, backed by models "
        "trained on real public datasets (see the project README for "
        "each model's dataset source and validated accuracy)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model(name: str):
    if models.get(name) is None:
        raise HTTPException(
            status_code=503,
            detail=f"The {name} model isn't trained yet. Run "
                   f"`python -m src.{name}_recommendation.train` "
                   f"(or the matching module for '{name}') and restart the API.",
        )
    return models[name]


@app.get("/health")
def health():
    """Reports which models are loaded and ready."""
    return {"status": "ok", "models_loaded": {k: v is not None for k, v in models.items()}}


@app.post("/predict/crop")
def predict_crop(request: CropRecommendationRequest):
    recommender = _require_model("crop")
    result = recommender.recommend(request.model_dump())
    return result


@app.post("/predict/fertilizer")
def predict_fertilizer(request: FertilizerRecommendationRequest):
    recommender = _require_model("fertilizer")
    result = recommender.recommend(request.model_dump(by_alias=True))
    return result


@app.post("/predict/irrigation")
def predict_irrigation(request: IrrigationPredictionRequest):
    predictor = _require_model("irrigation")
    result = predictor.predict(request.model_dump(by_alias=True))
    return result


@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    """Accepts a leaf image (JPG/PNG) and returns the predicted crop + condition."""
    detector = _require_model("disease")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload.jpg").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = detector.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.get("/iot/simulate")
def iot_simulate():
    """Returns one simulated raw sensor reading (see src/iot/README.md —
    this is synthetic by design, the paper's IoT layer has no real
    deployment to pull data from)."""
    sim = IoTSensorSimulator()
    return sim.read().as_dict()


@app.post("/iot/field-report")
def iot_field_report(request: FieldReportRequest = FieldReportRequest()):
    """Simulates one sensor reading and runs it through every trained
    tabular model, returning a combined report. Disease detection isn't
    included here since it needs an image, not a sensor reading — use
    POST /predict/disease directly for that."""
    sim = IoTSensorSimulator()
    reading = sim.read()

    report: dict = {"sensor_reading": reading.as_dict()}

    if models["crop"] is not None:
        report["crop_recommendation"] = models["crop"].recommend(
            to_crop_recommendation_input(reading)
        )
    else:
        report["crop_recommendation"] = {"error": "model not trained yet"}

    if models["fertilizer"] is not None:
        report["fertilizer_recommendation"] = models["fertilizer"].recommend(
            to_fertilizer_recommendation_input(reading, request.soil_type, request.crop_type)
        )
    else:
        report["fertilizer_recommendation"] = {"error": "model not trained yet"}

    if models["irrigation"] is not None:
        report["irrigation_prediction"] = models["irrigation"].predict(
            to_irrigation_prediction_input(reading, request.irrigation_crop_code, request.crop_days)
        )
    else:
        report["irrigation_prediction"] = {"error": "model not trained yet"}

    return JSONResponse(content=report)


@app.get("/")
def root():
    return {
        "message": "Smart Agriculture Assistant API",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/predict/crop",
            "/predict/fertilizer",
            "/predict/irrigation",
            "/predict/disease",
            "/iot/simulate",
            "/iot/field-report",
        ],
    }
