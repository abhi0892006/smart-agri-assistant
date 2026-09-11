"""
Dataset loading, validation, cleaning, and splitting for the
Crop Recommendation module.

Expects a CSV at config.CROP_REC_CSV with columns:
    N, P, K, temperature, humidity, ph, rainfall, label

This matches the paper's stated inputs (Section III-B): soil N/P/K,
pH, temperature, humidity, rainfall -> recommended crop (multi-class).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.common.preprocessing import impute_missing, clip_outliers_iqr, validate_columns


@dataclass
class CropRecDataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]


def load_raw_csv(csv_path=None) -> pd.DataFrame:
    """Load the raw crop recommendation CSV and validate its schema."""
    csv_path = csv_path or config.CROP_REC_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Crop recommendation dataset not found at {csv_path}.\n"
            "Download the dataset (e.g. the Kaggle 'Crop Recommendation "
            "Dataset', columns: N,P,K,temperature,humidity,ph,rainfall,label) "
            f"and place it at that path."
        )
    df = pd.read_csv(csv_path)
    validate_columns(df, config.CROP_REC_FEATURES + [config.CROP_REC_LABEL_COL])
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared imputation + outlier clipping to the feature columns."""
    df = df.copy()
    feature_cols = config.CROP_REC_FEATURES
    df[feature_cols] = impute_missing(df[feature_cols], strategy="median")
    df = clip_outliers_iqr(df, feature_cols)
    return df


def prepare_splits(df: pd.DataFrame | None = None, cfg: dict | None = None) -> CropRecDataSplits:
    """Clean, encode, scale, and split the dataset into train/val/test.

    Splitting is stratified on the label to keep class balance across
    splits, important given some public crop datasets have modest
    per-class sample counts.
    """
    cfg = cfg or config.CROP_REC_TRAIN_CONFIG
    if df is None:
        df = load_raw_csv()
    df = clean_dataframe(df)

    feature_cols = config.CROP_REC_FEATURES
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[config.CROP_REC_LABEL_COL].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    test_size = cfg["test_split"]
    val_size = cfg["val_split"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=cfg["random_seed"], stratify=y
    )
    # val_size is expressed as a fraction of the *original* dataset;
    # convert it to a fraction of the remaining train_val split.
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=cfg["random_seed"],
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return CropRecDataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_names=feature_cols,
    )
