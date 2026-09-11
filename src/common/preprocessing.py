"""
Shared preprocessing utilities for tabular agricultural data.

The paper explicitly calls out that sensor-derived (and even static
public-dataset) tabular data needs handling for: missing values, outliers,
and inconsistent ranges before being fed to the models (Section III-F).
This module centralizes that logic so the crop, fertilizer, and irrigation
pipelines all apply it consistently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def impute_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing numeric values.

    Args:
        df: input dataframe (numeric columns only should be passed for
            'median'/'mean'; non-numeric columns are left untouched).
        strategy: 'median' or 'mean'.

    Returns:
        A new dataframe with missing values imputed.
    """
    if strategy not in {"median", "mean"}:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
    return df


def clip_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """Clip outliers in the given numeric columns using the IQR rule.

    This is a conservative choice (clip, not drop) so we don't lose rows
    for a task where labeled agricultural data is already limited.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if expected columns are missing from a dataset.

    Fails loudly and early rather than letting a KeyError surface deep
    inside training or, worse, silently misaligning features at inference.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def validate_ranges(values: dict[str, float], expected_ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Check a single reading (e.g. from the IoT layer or a manual input)
    against expected sane ranges. Returns a list of warning strings for
    any field outside its expected range; does not raise, since a slightly
    out-of-range sensor reading is a warning, not necessarily invalid data.
    """
    warnings = []
    for key, (lo, hi) in expected_ranges.items():
        if key in values and values[key] is not None:
            v = values[key]
            if v < lo or v > hi:
                warnings.append(f"'{key}' value {v} is outside expected range [{lo}, {hi}]")
    return warnings
