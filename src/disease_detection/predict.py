"""
Inference for the Disease Detection module.

Loads the trained CNN and exposes `predict_disease(image_path)` for
other modules (or the future API layer) to call.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import torch
from PIL import Image
from torchvision import transforms

import config
from src.disease_detection.model import DiseaseDetectionCNN


class DiseaseDetector:
    def __init__(self, model_path=None, class_names_path=None, device: str | None = None):
        model_path = model_path or config.DISEASE_MODEL_PATH
        class_names_path = class_names_path or config.DISEASE_CLASS_NAMES_PATH

        for p in (model_path, class_names_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {p}. Run "
                    "`python -m src.disease_detection.train` first."
                )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = joblib.load(class_names_path)
        self.image_size = checkpoint["image_size"]

        self.model = DiseaseDetectionCNN(
            num_classes=checkpoint["num_classes"], image_size=self.image_size
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str, top_k: int = 3) -> dict:
        """Predict crop + disease for a single leaf image.

        Args:
            image_path: path to a JPG/PNG image of a leaf.
            top_k: number of top candidates to return.

        Returns:
            dict with 'top_prediction' (e.g. 'Tomato___Early_blight'),
            'crop' and 'condition' (parsed from the class name), 'top_k',
            and 'warnings' (empty list \u2014 reserved for future range/quality
            checks on the input image).
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_k = min(top_k, len(probs))
        top_indices = probs.argsort()[::-1][:top_k]
        top_predictions = [(self.class_names[i], float(probs[i])) for i in top_indices]

        top_class = top_predictions[0][0]
        crop, condition = top_class.split("___", 1) if "___" in top_class else (top_class, "unknown")

        return {
            "top_prediction": top_class,
            "crop": crop,
            "condition": "healthy" if condition == "healthy" else condition.replace("_", " "),
            "top_k": top_predictions,
            "warnings": [],
        }

    def valid_classes(self) -> list[str]:
        return list(self.class_names)


if __name__ == "__main__":
    import sys as _sys

    detector = DiseaseDetector()
    print("Classes this model recognizes:", detector.valid_classes())

    # Smoke test against one of the training images (real inference test).
    sample_dir = config.DISEASE_IMAGES_DIR / "Tomato___Early_blight"
    sample_images = list(sample_dir.glob("*.JPG")) + list(sample_dir.glob("*.jpg"))
    if sample_images:
        result = detector.predict(str(sample_images[0]))
        print(f"\nSample prediction on {sample_images[0].name}:")
        print(result)
    else:
        print("No sample image found for smoke test.")
