"""
Dataset loading, validation, cleaning, and splitting for the
Fertilizer Recommendation module.

Expects a CSV at config.FERTILIZER_CSV matching the public Kaggle
"Fertilizer Prediction" dataset schema:
    Temparature, Humidity, Moisture, Soil Type, Crop Type,
    Nitrogen, Potassium, Phosphorous, Fertilizer Name

This matches the paper's stated inputs (Section III-C): crop information
together with soil nutrient characteristics -> recommended fertilizer.
Soil Type and Crop Type are categorical and are one-hot encoded here;
the numeric columns are scaled the same way as the crop module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.common.preprocessing import impute_missing, clip_outliers_iqr, validate_columns


@dataclass
class FertilizerDataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    soil_encoder: OneHotEncoder
    crop_encoder: OneHotEncoder
    numeric_features: list[str]


def load_raw_csv(csv_path=None) -> pd.DataFrame:
    """Load the raw fertilizer CSV, normalize column names, and validate schema."""
    csv_path = csv_path or config.FERTILIZER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fertilizer dataset not found at {csv_path}.\n"
            "Download the dataset (Kaggle 'Fertilizer Prediction', columns: "
            "Temparature,Humidity,Moisture,Soil Type,Crop Type,Nitrogen,"
            f"Potassium,Phosphorous,Fertilizer Name) and place it at that path."
        )
    df = pd.read_csv(csv_path)
    # The public CSV ships with a trailing space in "Humidity " — normalize
    # all column names defensively so downstream code can rely on exact names.
    df.columns = [c.strip() for c in df.columns]
    required = (
        config.FERTILIZER_NUMERIC_FEATURES
        + config.FERTILIZER_CATEGORICAL_FEATURES
        + [config.FERTILIZER_LABEL_COL]
    )
    validate_columns(df, required)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared imputation + outlier clipping to numeric feature columns."""
    df = df.copy()
    numeric_cols = config.FERTILIZER_NUMERIC_FEATURES
    df[numeric_cols] = impute_missing(df[numeric_cols], strategy="median")
    df = clip_outliers_iqr(df, numeric_cols)
    return df


def prepare_splits(df: pd.DataFrame | None = None, cfg: dict | None = None) -> FertilizerDataSplits:
    """Clean, encode (numeric scale + categorical one-hot), and split the dataset."""
    cfg = cfg or config.FERTILIZER_TRAIN_CONFIG
    if df is None:
        df = load_raw_csv()
    df = clean_dataframe(df)

    numeric_cols = config.FERTILIZER_NUMERIC_FEATURES
    X_numeric = df[numeric_cols].values.astype(np.float32)

    soil_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_soil = soil_encoder.fit_transform(df[["Soil Type"]])

    crop_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_crop = crop_encoder.fit_transform(df[["Crop Type"]])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[config.FERTILIZER_LABEL_COL].values)

    test_size = cfg["test_split"]
    val_size = cfg["val_split"]

    # Stratification requires every class to have at least 2 samples in the
    # split; with only 99 rows across 7 classes this is checked explicitly
    # so a confusing sklearn error doesn't surface instead.
    class_counts = pd.Series(y).value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "At least one fertilizer class has fewer than 2 samples, which "
            "breaks stratified splitting. Check the dataset for very rare classes."
        )

    indices = np.arange(len(y))
    idx_train_val, idx_test = train_test_split(
        indices, test_size=test_size, random_state=cfg["random_seed"], stratify=y
    )
    relative_val_size = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val_size,
        random_state=cfg["random_seed"],
        stratify=y[idx_train_val],
    )

    def build_features(idx):
        return np.hstack([X_numeric[idx], X_soil[idx], X_crop[idx]]).astype(np.float32)

    X_train_raw = build_features(idx_train)
    X_val_raw = build_features(idx_val)
    X_test_raw = build_features(idx_test)

    # Only scale the numeric portion; one-hot columns are already 0/1.
    n_numeric = len(numeric_cols)
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()

    X_train[:, :n_numeric] = scaler.fit_transform(X_train_raw[:, :n_numeric])
    X_val[:, :n_numeric] = scaler.transform(X_val_raw[:, :n_numeric])
    X_test[:, :n_numeric] = scaler.transform(X_test_raw[:, :n_numeric])

    return FertilizerDataSplits(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y[idx_train],
        y_val=y[idx_val],
        y_test=y[idx_test],
        scaler=scaler,
        label_encoder=label_encoder,
        soil_encoder=soil_encoder,
        crop_encoder=crop_encoder,
        numeric_features=numeric_cols,
    )
