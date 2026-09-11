"""
Training entry point for the Disease Detection module.

Usage:
    python -m src.disease_detection.train
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.common.metrics import evaluate_classification
from src.disease_detection.dataset import prepare_splits
from src.disease_detection.model import DiseaseDetectionCNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    cfg = config.DISEASE_TRAIN_CONFIG
    set_seed(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing image data...")
    splits = prepare_splits(cfg=cfg)
    print(f"Classes ({len(splits.class_names)}): {splits.class_names}")
    print(
        f"Train/val/test sizes: {len(splits.train_dataset)}/"
        f"{len(splits.val_dataset)}/{len(splits.test_dataset)}"
    )

    train_loader = DataLoader(splits.train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(splits.val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(splits.test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = DiseaseDetectionCNN(
        num_classes=len(splits.class_names), image_size=config.DISEASE_IMAGE_SIZE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

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

    y_pred, y_true = predict_all(model, test_loader, device)
    result = evaluate_classification(y_true, y_pred, class_names=splits.class_names)

    print("\n=== Test set performance ===")
    print(f"(test set size: {len(y_true)} images across {len(splits.class_names)} classes; "
          "this is a small subset of the full PlantVillage dataset \u2014 see "
          "data/disease_images/README.md for the scope caveat)")
    print(result.summary())

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(splits.class_names),
            "image_size": config.DISEASE_IMAGE_SIZE,
        },
        config.DISEASE_MODEL_PATH,
    )
    joblib.dump(splits.class_names, config.DISEASE_CLASS_NAMES_PATH)

    print(f"\nSaved model to {config.DISEASE_MODEL_PATH}")
    print(f"Saved class names to {config.DISEASE_CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
