"""
eval.py — Evaluation module for coral reef classification.

Generates:
    - Confusion matrix (PNG)
    - Classification report (precision, recall, F1)
    - Per-class accuracy bar chart

CLI:
    python -m src.eval --model-type dl --checkpoint models/best_model.pth
    python -m src.eval --model-type ml --checkpoint models/random_forest.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_dataloaders
from src.train_ml import extract_features
from src.train_resnet import build_model
from src.utils import ensure_dir, get_device, load_config, setup_logger

logger = setup_logger("eval")


# ================================================================
#  Deep-learning evaluation
# ================================================================

def evaluate_dl(checkpoint_path: Path, cfg: dict) -> None:
    """Load a DL checkpoint and evaluate on the test split."""
    device = get_device(cfg["training"].get("device", "auto"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", cfg["training"]["model"])
    num_classes = ckpt.get("num_classes", cfg["dataset"]["num_classes"])

    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(cfg)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    _report_and_plot(np.array(all_labels), np.array(all_preds), cfg, tag=model_name)


# ================================================================
#  Traditional-ML evaluation
# ================================================================

def evaluate_ml(checkpoint_path: Path, cfg: dict) -> None:
    """Load a sklearn/XGB model and evaluate on processed images."""
    model = joblib.load(checkpoint_path)
    scaler_path = Path(cfg["paths"]["models"]) / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    class_names = cfg["dataset"]["classes"]
    feature_list = cfg["traditional_ml"].get("features", ["color_histogram", "haralick", "hog"])
    data_dir = Path(cfg["paths"]["processed_data"])

    X_list, y_list = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for idx, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            X_list.append(extract_features(img, feature_list))
            y_list.append(idx)

    X = np.array(X_list)
    y = np.array(y_list)
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    tag = checkpoint_path.stem
    _report_and_plot(y, preds, cfg, tag=tag)


# ================================================================
#  Shared reporting
# ================================================================

def _report_and_plot(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: dict, tag: str = "model"
) -> None:
    class_names = cfg["dataset"]["classes"]
    runs_dir = ensure_dir(cfg["paths"]["runs"])

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\n📊  Classification Report (%s):\n%s", tag, report)

    report_path = runs_dir / f"classification_report_{tag}.txt"
    with open(report_path, "w") as fh:
        fh.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {tag}")
    cm_path = runs_dir / f"confusion_matrix_{tag}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", cm_path)

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(class_names, per_class_acc, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Per-class Accuracy — {tag}")
    for bar, acc in zip(bars, per_class_acc):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{acc:.2%}", ha="center", fontsize=10)
    acc_path = runs_dir / f"per_class_accuracy_{tag}.png"
    fig2.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved per-class accuracy chart → %s", acc_path)


# ================================================================
#  CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coral reef classifier")
    parser.add_argument("--model-type", choices=["dl", "ml"], required=True,
                        help="Deep learning or traditional ML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth or .joblib)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)

    if args.model_type == "dl":
        evaluate_dl(ckpt, cfg)
    else:
        evaluate_ml(ckpt, cfg)


if __name__ == "__main__":
    main()
