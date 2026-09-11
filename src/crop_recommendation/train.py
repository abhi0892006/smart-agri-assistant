"""
Training entry point for the Crop Recommendation module.

Usage:
    python -m src.crop_recommendation.train

Trains the baseline MLP, evaluates it on the held-out test split with the
metric set the paper specifies (accuracy, precision, recall, F1, confusion
matrix), and persists the model + scaler + label encoder to disk so
predict.py can load them for inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.common.metrics import evaluate_classification
from src.crop_recommendation.dataset import prepare_splits
from src.crop_recommendation.model import CropRecommendationMLP


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_labels(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    preds = torch.argmax(logits, dim=1)
    return preds.cpu().numpy()


def main():
    cfg = config.CROP_REC_TRAIN_CONFIG
    set_seed(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    splits = prepare_splits(cfg=cfg)

    num_classes = len(splits.label_encoder.classes_)
    input_dim = splits.X_train.shape[1]
    print(f"Input dim: {input_dim}, classes: {num_classes}")

    model = CropRecommendationMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    train_loader = make_loader(splits.X_train, splits.y_train, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(splits.X_val, splits.y_val, cfg["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on held-out test split
    y_pred = predict_labels(model, splits.X_test, device)
    class_names = list(splits.label_encoder.classes_)
    result = evaluate_classification(splits.y_test, y_pred, class_names=class_names)

    print("\n=== Test set performance ===")
    print(result.summary())

    # Persist artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": cfg["hidden_dims"],
            "dropout": cfg["dropout"],
        },
        config.CROP_REC_MODEL_PATH,
    )
    joblib.dump(splits.scaler, config.CROP_REC_SCALER_PATH)
    joblib.dump(splits.label_encoder, config.CROP_REC_LABEL_ENCODER_PATH)

    print(f"\nSaved model to {config.CROP_REC_MODEL_PATH}")
    print(f"Saved scaler to {config.CROP_REC_SCALER_PATH}")
    print(f"Saved label encoder to {config.CROP_REC_LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
