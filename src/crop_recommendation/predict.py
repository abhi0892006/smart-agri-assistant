"""
Inference for the Crop Recommendation module.

Loads the trained model + scaler + label encoder and exposes a single
`recommend_crop(...)` function that other modules (e.g. the future API
layer or the IoT ingestion pipeline) can call directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import torch

import config
from src.common.preprocessing import validate_ranges
from src.crop_recommendation.model import CropRecommendationMLP

# Sane expected ranges for each input, used only to emit warnings for
# clearly implausible sensor/manual-entry values (Section III-F: sensor
# readings can be noisy or drift).
EXPECTED_RANGES = {
    "N": (0, 200),
    "P": (0, 200),
    "K": (0, 250),
    "temperature": (-10, 55),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 3500),
}


class CropRecommender:
    def __init__(
        self,
        model_path=None,
        scaler_path=None,
        label_encoder_path=None,
        device: str | None = None,
    ):
        model_path = model_path or config.CROP_REC_MODEL_PATH
        scaler_path = scaler_path or config.CROP_REC_SCALER_PATH
        label_encoder_path = label_encoder_path or config.CROP_REC_LABEL_ENCODER_PATH

        for p in (model_path, scaler_path, label_encoder_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {p}. Run "
                    "`python -m src.crop_recommendation.train` first."
                )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CropRecommendationMLP(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def recommend(self, reading: dict, top_k: int = 3) -> dict:
        """Recommend crop(s) for a single soil/environment reading.

        Args:
            reading: dict with keys N, P, K, temperature, humidity, ph, rainfall.
            top_k: number of top candidate crops to return with confidence.

        Returns:
            dict with 'top_prediction', 'top_k' (list of (crop, probability)),
            and 'warnings' (any out-of-range inputs detected).
        """
        missing = [f for f in config.CROP_REC_FEATURES if f not in reading]
        if missing:
            raise ValueError(f"Missing required fields for prediction: {missing}")

        warnings = validate_ranges(reading, EXPECTED_RANGES)

        x = np.array([[reading[f] for f in config.CROP_REC_FEATURES]], dtype=np.float32)
        x_scaled = self.scaler.transform(x).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x_scaled).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_crops = [
            (self.label_encoder.inverse_transform([i])[0], float(probs[i]))
            for i in top_indices
        ]

        return {
            "top_prediction": top_crops[0][0],
            "top_k": top_crops,
            "warnings": warnings,
        }


if __name__ == "__main__":
    # Simple manual smoke test (requires a trained model on disk).
    recommender = CropRecommender()
    sample_reading = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    result = recommender.recommend(sample_reading)
    print(result)
