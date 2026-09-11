"""
Inference for the Irrigation Prediction module.

Loads the trained model + scaler + crop encoder and exposes
`predict_irrigation(...)` for other modules to call.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch

import config
from src.common.preprocessing import validate_ranges
from src.irrigation_prediction.model import IrrigationPredictionMLP

EXPECTED_RANGES = {
    "CropDays": (0, 365),
    "Soil Moisture": (0, 1000),
    "Soil Temperature": (0, 60),
    "Temperature": (0, 55),
    "Humidity": (0, 100),
}


class IrrigationPredictor:
    def __init__(
        self,
        model_path=None,
        scaler_path=None,
        crop_encoder_path=None,
        device: str | None = None,
    ):
        model_path = model_path or config.IRRIGATION_MODEL_PATH
        scaler_path = scaler_path or config.IRRIGATION_SCALER_PATH
        crop_encoder_path = crop_encoder_path or config.IRRIGATION_CROP_ENCODER_PATH

        for p in (model_path, scaler_path, crop_encoder_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {p}. Run "
                    "`python -m src.irrigation_prediction.train` first."
                )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = IrrigationPredictionMLP(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.crop_encoder = joblib.load(crop_encoder_path)

    def valid_crop_types(self) -> list[int]:
        """Returns the CropType codes seen during training (see
        config.IRRIGATION_CROP_TYPE_MAP for the code -> name mapping)."""
        return list(self.crop_encoder.categories_[0])

    def predict(self, reading: dict) -> dict:
        """Predict whether irrigation is needed for a single reading.

        Args:
            reading: dict with keys CropDays, Soil Moisture, Soil
                Temperature, Temperature, Humidity (numeric), plus
                'CropType' (int code, see config.IRRIGATION_CROP_TYPE_MAP).

        Returns:
            dict with 'irrigate' (bool), 'confidence' (float, probability
            of the predicted class), 'probabilities' ({'No': .., 'Yes': ..}),
            and 'warnings'.
        """
        missing = [
            f for f in config.IRRIGATION_NUMERIC_FEATURES + config.IRRIGATION_CATEGORICAL_FEATURES
            if f not in reading
        ]
        if missing:
            raise ValueError(f"Missing required fields for prediction: {missing}")

        warnings = validate_ranges(reading, EXPECTED_RANGES)

        if reading["CropType"] not in self.valid_crop_types():
            warnings.append(
                f"CropType={reading['CropType']!r} was not seen during training "
                f"(known codes: {self.valid_crop_types()}, see config.IRRIGATION_CROP_TYPE_MAP); "
                "prediction may be unreliable."
            )

        x_numeric = np.array(
            [[reading[f] for f in config.IRRIGATION_NUMERIC_FEATURES]], dtype=np.float32
        )
        x_numeric_scaled = self.scaler.transform(x_numeric)

        x_crop = self.crop_encoder.transform(
            pd.DataFrame([[reading["CropType"]]], columns=["CropType"])
        )

        x = np.hstack([x_numeric_scaled, x_crop]).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        return {
            "irrigate": bool(predicted_class == 1),
            "confidence": float(probs[predicted_class]),
            "probabilities": {"No": float(probs[0]), "Yes": float(probs[1])},
            "warnings": warnings,
        }


if __name__ == "__main__":
    predictor = IrrigationPredictor()
    print("Crop type codes:", config.IRRIGATION_CROP_TYPE_MAP)

    samples = [
        {"CropType": 1, "CropDays": 3, "Soil Moisture": 230, "Soil Temperature": 25, "Temperature": 30, "Humidity": 60},
        {"CropType": 2, "CropDays": 50, "Soil Moisture": 800, "Soil Temperature": 22, "Temperature": 28, "Humidity": 55},
    ]
    for s in samples:
        print(s, "->", predictor.predict(s))
