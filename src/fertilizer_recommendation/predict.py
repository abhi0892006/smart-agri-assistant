"""
Inference for the Fertilizer Recommendation module.

Loads the trained model + scaler + encoders and exposes
`recommend_fertilizer(...)` for other modules to call.
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
from src.fertilizer_recommendation.model import FertilizerRecommendationMLP

EXPECTED_RANGES = {
    "Temparature": (0, 55),
    "Humidity": (0, 100),
    "Moisture": (0, 100),
    "Nitrogen": (0, 150),
    "Potassium": (0, 150),
    "Phosphorous": (0, 150),
}


class FertilizerRecommender:
    def __init__(
        self,
        model_path=None,
        scaler_path=None,
        label_encoder_path=None,
        soil_encoder_path=None,
        crop_encoder_path=None,
        device: str | None = None,
    ):
        model_path = model_path or config.FERTILIZER_MODEL_PATH
        scaler_path = scaler_path or config.FERTILIZER_SCALER_PATH
        label_encoder_path = label_encoder_path or config.FERTILIZER_LABEL_ENCODER_PATH
        soil_encoder_path = soil_encoder_path or config.FERTILIZER_SOIL_ENCODER_PATH
        crop_encoder_path = crop_encoder_path or config.FERTILIZER_CROP_ENCODER_PATH

        for p in (model_path, scaler_path, label_encoder_path, soil_encoder_path, crop_encoder_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {p}. Run "
                    "`python -m src.fertilizer_recommendation.train` first."
                )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FertilizerRecommendationMLP(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.soil_encoder = joblib.load(soil_encoder_path)
        self.crop_encoder = joblib.load(crop_encoder_path)

    def valid_soil_types(self) -> list[str]:
        return list(self.soil_encoder.categories_[0])

    def valid_crop_types(self) -> list[str]:
        return list(self.crop_encoder.categories_[0])

    def recommend(self, reading: dict, top_k: int = 3) -> dict:
        """Recommend fertilizer(s) for a single crop/soil reading.

        Args:
            reading: dict with keys Temparature, Humidity, Moisture,
                Nitrogen, Potassium, Phosphorous (numeric), plus
                'Soil Type' and 'Crop Type' (strings, must match training
                categories — see valid_soil_types()/valid_crop_types()).
            top_k: number of top candidate fertilizers to return.

        Returns:
            dict with 'top_prediction', 'top_k', and 'warnings'.
        """
        missing = [
            f for f in config.FERTILIZER_NUMERIC_FEATURES + config.FERTILIZER_CATEGORICAL_FEATURES
            if f not in reading
        ]
        if missing:
            raise ValueError(f"Missing required fields for prediction: {missing}")

        warnings = validate_ranges(reading, EXPECTED_RANGES)

        if reading["Soil Type"] not in self.valid_soil_types():
            warnings.append(
                f"'Soil Type'={reading['Soil Type']!r} was not seen during training "
                f"(known: {self.valid_soil_types()}); prediction may be unreliable."
            )
        if reading["Crop Type"] not in self.valid_crop_types():
            warnings.append(
                f"'Crop Type'={reading['Crop Type']!r} was not seen during training "
                f"(known: {self.valid_crop_types()}); prediction may be unreliable."
            )

        x_numeric = np.array(
            [[reading[f] for f in config.FERTILIZER_NUMERIC_FEATURES]], dtype=np.float32
        )
        x_numeric_scaled = self.scaler.transform(x_numeric)

        import pandas as pd  # local import: only needed here, keeps module load light

        x_soil = self.soil_encoder.transform(pd.DataFrame([[reading["Soil Type"]]], columns=["Soil Type"]))
        x_crop = self.crop_encoder.transform(pd.DataFrame([[reading["Crop Type"]]], columns=["Crop Type"]))

        x = np.hstack([x_numeric_scaled, x_soil, x_crop]).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_fertilizers = [
            (self.label_encoder.inverse_transform([i])[0], float(probs[i]))
            for i in top_indices
        ]

        return {
            "top_prediction": top_fertilizers[0][0],
            "top_k": top_fertilizers,
            "warnings": warnings,
        }


if __name__ == "__main__":
    recommender = FertilizerRecommender()
    sample_reading = {
        "Temparature": 26,
        "Humidity": 52,
        "Moisture": 38,
        "Soil Type": "Sandy",
        "Crop Type": "Maize",
        "Nitrogen": 37,
        "Potassium": 0,
        "Phosphorous": 0,
    }
    result = recommender.recommend(sample_reading)
    print(result)
