"""
Shared evaluation utilities.

Implements the metric set the paper specifies in Section V:
accuracy, precision, recall, F1-score, confusion matrix for classification
tasks, and MAE, MSE, RMSE, R^2 for regression tasks (irrigation, if
formulated as regression).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class ClassificationResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray
    report: str
    class_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision (macro): {self.precision_macro:.4f}",
            f"Recall (macro):    {self.recall_macro:.4f}",
            f"F1 (macro):        {self.f1_macro:.4f}",
            "",
            "Classification report:",
            self.report,
        ]
        return "\n".join(lines)


@dataclass
class RegressionResult:
    mae: float
    mse: float
    rmse: float
    r2: float

    def summary(self) -> str:
        return (
            f"MAE:  {self.mae:.4f}\n"
            f"MSE:  {self.mse:.4f}\n"
            f"RMSE: {self.rmse:.4f}\n"
            f"R^2:  {self.r2:.4f}"
        )


def evaluate_classification(y_true, y_pred, class_names: list[str] | None = None) -> ClassificationResult:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    return ClassificationResult(
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        confusion=cm,
        report=report,
        class_names=class_names or [],
    )


def evaluate_regression(y_true, y_pred) -> RegressionResult:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return RegressionResult(mae=mae, mse=mse, rmse=rmse, r2=r2)
