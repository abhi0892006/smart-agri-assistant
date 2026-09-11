"""
Dataset loading, validation, cleaning, and splitting for the
Irrigation Prediction module.

Expects a CSV at config.IRRIGATION_CSV matching the GCEK IoT Community
"Irrigation Dataset" schema:
    CropType, CropDays, Soil Moisture, Soil Temperature, Temperature,
    Humidity, Irrigation(Y/N)

This matches the paper's stated inputs for this task (crop, soil, and
environmental readings -> irrigation requirement), framed here as binary
classification since that's how the real dataset labels it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.common.preprocessing import impute_missing, clip_outliers_iqr, validate_columns


@dataclass
class IrrigationDataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    crop_encoder: OneHotEncoder
    numeric_features: list[str]
    class_weights: np.ndarray  # for the weighted loss, computed from y_train


def load_raw_csv(csv_path=None) -> pd.DataFrame:
    """Load the raw irrigation CSV and validate its schema."""
    csv_path = csv_path or config.IRRIGATION_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Irrigation dataset not found at {csv_path}.\n"
            "Expected columns: CropType, CropDays, Soil Moisture, "
            "Soil Temperature, Temperature, Humidity, Irrigation(Y/N)."
        )
    df = pd.read_csv(csv_path)
    required = (
        config.IRRIGATION_NUMERIC_FEATURES
        + config.IRRIGATION_CATEGORICAL_FEATURES
        + [config.IRRIGATION_LABEL_COL]
    )
    validate_columns(df, required)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared imputation + outlier clipping to numeric feature columns."""
    df = df.copy()
    numeric_cols = config.IRRIGATION_NUMERIC_FEATURES
    df[numeric_cols] = impute_missing(df[numeric_cols], strategy="median")
    df = clip_outliers_iqr(df, numeric_cols)
    return df


def prepare_splits(df: pd.DataFrame | None = None, cfg: dict | None = None) -> IrrigationDataSplits:
    """Clean, encode, scale, and split the dataset into train/val/test.

    Uses stratified splitting on the label (important given the ~83/17
    class imbalance) and computes inverse-frequency class weights from
    the training split for use in a weighted loss function.
    """
    cfg = cfg or config.IRRIGATION_TRAIN_CONFIG
    if df is None:
        df = load_raw_csv()
    df = clean_dataframe(df)

    numeric_cols = config.IRRIGATION_NUMERIC_FEATURES
    X_numeric = df[numeric_cols].values.astype(np.float32)

    crop_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_crop = crop_encoder.fit_transform(df[["CropType"]])

    y = df[config.IRRIGATION_LABEL_COL].values.astype(np.int64)

    test_size = cfg["test_split"]
    val_size = cfg["val_split"]

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
        return np.hstack([X_numeric[idx], X_crop[idx]]).astype(np.float32)

    X_train_raw = build_features(idx_train)
    X_val_raw = build_features(idx_val)
    X_test_raw = build_features(idx_test)

    n_numeric = len(numeric_cols)
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()

    X_train[:, :n_numeric] = scaler.fit_transform(X_train_raw[:, :n_numeric])
    X_val[:, :n_numeric] = scaler.transform(X_val_raw[:, :n_numeric])
    X_test[:, :n_numeric] = scaler.transform(X_test_raw[:, :n_numeric])

    y_train = y[idx_train]
    class_counts = np.bincount(y_train)
    # Inverse-frequency weighting: rarer class gets a higher weight so the
    # loss doesn't let the model coast by always predicting the majority class.
    class_weights = len(y_train) / (len(class_counts) * class_counts)

    return IrrigationDataSplits(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train,
        y_val=y[idx_val],
        y_test=y[idx_test],
        scaler=scaler,
        crop_encoder=crop_encoder,
        numeric_features=numeric_cols,
        class_weights=class_weights.astype(np.float32),
    )
