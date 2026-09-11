"""
Central configuration for the Smart Agriculture Assistant project.

All paths are resolved relative to the project root so scripts work
regardless of the current working directory they're invoked from.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CROP_REC_DATA_DIR = DATA_DIR / "crop_recommendation"
FERTILIZER_DATA_DIR = DATA_DIR / "fertilizer"
IRRIGATION_DATA_DIR = DATA_DIR / "irrigation"
DISEASE_IMAGES_DIR = DATA_DIR / "disease_images"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Crop Recommendation
# ---------------------------------------------------------------------------
# Expected CSV columns (matches the widely-used Kaggle "Crop Recommendation
# Dataset" schema, which lines up with the paper's stated inputs:
# N, P, K, temperature, humidity, ph, rainfall -> label).
CROP_REC_CSV = CROP_REC_DATA_DIR / "crop_recommendation.csv"
CROP_REC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_REC_LABEL_COL = "label"

CROP_REC_MODEL_PATH = MODELS_DIR / "crop_recommendation_mlp.pt"
CROP_REC_SCALER_PATH = MODELS_DIR / "crop_recommendation_scaler.joblib"
CROP_REC_LABEL_ENCODER_PATH = MODELS_DIR / "crop_recommendation_label_encoder.joblib"

# Training hyperparameters (baseline MLP, per the paper's description of
# a baseline architecture for tabular tasks).
CROP_REC_TRAIN_CONFIG = {
    "hidden_dims": [64, 32],
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "early_stopping_patience": 10,
}

# ---------------------------------------------------------------------------
# Fertilizer Recommendation
# ---------------------------------------------------------------------------
# Expected CSV columns (matches the public Kaggle "Fertilizer Prediction"
# dataset, which lines up with the paper's stated inputs for this task:
# crop + soil nutrient characteristics -> fertilizer). Note the CSV's
# original header has a trailing space after "Humidity" and inconsistent
# capitalization ("Temparature") — dataset.py normalizes these on load.
FERTILIZER_CSV = FERTILIZER_DATA_DIR / "fertilizer_prediction.csv"
FERTILIZER_NUMERIC_FEATURES = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERTILIZER_CATEGORICAL_FEATURES = ["Soil Type", "Crop Type"]
FERTILIZER_LABEL_COL = "Fertilizer Name"

FERTILIZER_MODEL_PATH = MODELS_DIR / "fertilizer_recommendation_mlp.pt"
FERTILIZER_SCALER_PATH = MODELS_DIR / "fertilizer_recommendation_scaler.joblib"
FERTILIZER_LABEL_ENCODER_PATH = MODELS_DIR / "fertilizer_recommendation_label_encoder.joblib"
FERTILIZER_SOIL_ENCODER_PATH = MODELS_DIR / "fertilizer_recommendation_soil_encoder.joblib"
FERTILIZER_CROP_ENCODER_PATH = MODELS_DIR / "fertilizer_recommendation_crop_encoder.joblib"

# The public dataset for this task is small (99 rows, 7 classes), so the
# model is more prone to overfitting than crop recommendation's 2200-row
# dataset. Config reflects that: smaller network, stronger regularization,
# and a larger relative validation/test split.
FERTILIZER_TRAIN_CONFIG = {
    "hidden_dims": [32, 16],
    "dropout": 0.3,
    "batch_size": 8,
    "epochs": 150,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "test_split": 0.2,
    "random_seed": 42,
    "early_stopping_patience": 15,
}

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Irrigation Prediction
# ---------------------------------------------------------------------------
# Expects a CSV matching the GCEK IoT Community "Irrigation Dataset" schema:
#   CropType, CropDays, Soil Moisture, Soil Temperature, Temperature,
#   Humidity, Irrigation(Y/N)
# CropType is coded 1=Paddy, 2=Ground Nuts in the source dataset.
# Framed as BINARY CLASSIFICATION (irrigate: yes/no), matching what this
# real dataset actually labels — the paper left classification vs.
# regression open, and this is the real-world framing that public data
# supports.
IRRIGATION_CSV = IRRIGATION_DATA_DIR / "irrigation_prediction.csv"
IRRIGATION_NUMERIC_FEATURES = ["CropDays", "Soil Moisture", "Soil Temperature", "Temperature", "Humidity"]
IRRIGATION_CATEGORICAL_FEATURES = ["CropType"]
IRRIGATION_LABEL_COL = "Irrigation(Y/N)"
IRRIGATION_CROP_TYPE_MAP = {1: "Paddy", 2: "Ground Nuts"}

IRRIGATION_MODEL_PATH = MODELS_DIR / "irrigation_prediction_mlp.pt"
IRRIGATION_SCALER_PATH = MODELS_DIR / "irrigation_prediction_scaler.joblib"
IRRIGATION_CROP_ENCODER_PATH = MODELS_DIR / "irrigation_prediction_crop_encoder.joblib"

# Dataset is small (150 rows) AND imbalanced (~83% "No" / 17% "Yes"), so:
# - smaller network + strong dropout (overfitting risk, same reasoning as fertilizer)
# - class-weighted loss is computed at train time from the actual label
#   distribution rather than hardcoded here, so it stays correct if the
#   dataset changes.
IRRIGATION_TRAIN_CONFIG = {
    "hidden_dims": [32, 16],
    "dropout": 0.3,
    "batch_size": 8,
    "epochs": 150,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "test_split": 0.2,
    "random_seed": 42,
    "early_stopping_patience": 15,
}

# ---------------------------------------------------------------------------
# Disease Detection (CNN)
# ---------------------------------------------------------------------------
# data/disease_images/<ClassName>/*.jpg, ImageFolder-style. ClassName follows
# the PlantVillage naming convention "<Crop>___<Disease-or-healthy>".
#
# IMPORTANT: this is a REAL but DELIBERATELY SMALL subset of the public
# PlantVillage dataset (https://github.com/spMohanty/PlantVillage-Dataset,
# 54,306 images / 38 classes total). Only 8 classes (4 crops x
# disease/healthy), capped at 150 images/class (1200 images total), are
# included here so a CNN can train in a few minutes on CPU. This is a
# pipeline demonstration, not the full multi-crop model the paper
# describes — see data/disease_images/README.md for how to scale up.
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
]
DISEASE_IMAGE_SIZE = 96  # small on purpose: fast CPU training
DISEASE_MODEL_PATH = MODELS_DIR / "disease_detection_cnn.pt"
DISEASE_CLASS_NAMES_PATH = MODELS_DIR / "disease_detection_classes.joblib"

DISEASE_TRAIN_CONFIG = {
    "batch_size": 16,
    "epochs": 25,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "early_stopping_patience": 6,
}

# ---------------------------------------------------------------------------
# IoT Simulator
# ---------------------------------------------------------------------------
# The paper's IoT layer (soil moisture/temp/pH sensors, NPK sensor, air
# temp/humidity, rain gauge, light sensor -> ESP32/MQTT -> server) is
# explicitly conceptual: no real sensor deployment exists. This module is
# therefore INTENTIONALLY synthetic (unlike the other four, which all use
# real public datasets) — it generates physically-plausible readings with
# a diurnal (day/night) cycle, not random noise, so it's useful for
# exercising the trained models end-to-end.
IOT_SENSOR_RANGES = {
    "N": (10, 140),
    "P": (5, 145),
    "K": (5, 205),
    "ph": (4.5, 8.5),
    "rainfall_mm_per_day": (0, 300),
    "air_temperature_c": (15, 40),   # diurnal midpoint/amplitude applied on top
    "humidity_pct": (30, 95),
    # NOTE: the two source datasets measure "soil moisture" in genuinely
    # incompatible units (confirmed by inspecting both real datasets, not
    # assumed): fertilizer_prediction.csv's "Moisture" column is a 25-65
    # percentage-like scale, while irrigation_prediction.csv's "Soil
    # Moisture" is a raw sensor/ADC reading in the 100-900 range. Rather
    # than inventing an unfounded conversion formula between them, the
    # simulator generates both independently.
    "soil_moisture_pct": (20, 70),        # for fertilizer_recommendation
    "soil_moisture_raw": (100, 900),      # for irrigation_prediction
    "soil_temperature_c": (15, 40),
    "light_lux": (0, 100000),         # 0 at night, up to full sun at midday
}
IOT_RANDOM_SEED = 7
