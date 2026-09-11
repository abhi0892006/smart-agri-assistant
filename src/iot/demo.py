"""
End-to-end demo: simulate live sensor readings and feed them into the
trained crop, fertilizer, and irrigation models to produce a combined
"field report" — the closest thing to a working preview of the full
system before the API layer (Phase 6) exists.

Usage:
    python -m src.iot.demo

Requires Phases 1-3 to already be trained (run each module's train.py
first if the models/ artifacts aren't there yet).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.iot.simulator import IoTSensorSimulator
from src.iot.adapters import (
    to_crop_recommendation_input,
    to_fertilizer_recommendation_input,
    to_irrigation_prediction_input,
)
from src.crop_recommendation.predict import CropRecommender
from src.fertilizer_recommendation.predict import FertilizerRecommender
from src.irrigation_prediction.predict import IrrigationPredictor


def run_field_report(soil_type: str = "Sandy", crop_type: str = "Maize",
                      irrigation_crop_code: int = 1, crop_days: int = 15) -> None:
    """Simulates one sensor reading and runs it through all three trained
    tabular models, printing a combined report.

    Note: soil_type/crop_type (for fertilizer) and irrigation_crop_code
    (for irrigation) are scenario parameters you choose, not predicted —
    see src/iot/adapters.py for why (the three models don't share a crop
    naming convention).
    """
    sim = IoTSensorSimulator()
    reading = sim.read()

    print("=" * 60)
    print("SIMULATED SENSOR READING")
    print("=" * 60)
    for k, v in reading.as_dict().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION (predicted from raw sensor data)")
    print("=" * 60)
    try:
        crop_rec = CropRecommender()
        result = crop_rec.recommend(to_crop_recommendation_input(reading))
        print(f"  Top pick: {result['top_prediction']}")
        for crop, prob in result["top_k"]:
            print(f"    {crop}: {prob:.1%}")
        if result["warnings"]:
            print(f"  Warnings: {result['warnings']}")
    except FileNotFoundError as e:
        print(f"  [skipped] {e}")

    print(f"\n" + "=" * 60)
    print(f"FERTILIZER RECOMMENDATION (scenario: {soil_type} soil, {crop_type})")
    print("=" * 60)
    try:
        fert_rec = FertilizerRecommender()
        result = fert_rec.recommend(
            to_fertilizer_recommendation_input(reading, soil_type, crop_type)
        )
        print(f"  Top pick: {result['top_prediction']}")
        for fert, prob in result["top_k"]:
            print(f"    {fert}: {prob:.1%}")
        if result["warnings"]:
            print(f"  Warnings: {result['warnings']}")
    except FileNotFoundError as e:
        print(f"  [skipped] {e}")

    crop_name = config.IRRIGATION_CROP_TYPE_MAP.get(irrigation_crop_code, "unknown")
    print(f"\n" + "=" * 60)
    print(f"IRRIGATION PREDICTION (scenario: {crop_name}, day {crop_days})")
    print("=" * 60)
    try:
        irrigation = IrrigationPredictor()
        result = irrigation.predict(
            to_irrigation_prediction_input(reading, irrigation_crop_code, crop_days)
        )
        decision = "IRRIGATE" if result["irrigate"] else "no irrigation needed"
        print(f"  Decision: {decision} (confidence: {result['confidence']:.1%})")
        print(f"  Probabilities: {result['probabilities']}")
        if result["warnings"]:
            print(f"  Warnings: {result['warnings']}")
    except FileNotFoundError as e:
        print(f"  [skipped] {e}")

    print("\n" + "=" * 60)
    print("Note: disease detection isn't part of this report — it needs an")
    print("actual leaf photo as input, not a sensor reading. Use")
    print("src.disease_detection.predict.DiseaseDetector directly with an image.")
    print("=" * 60)


if __name__ == "__main__":
    run_field_report()
