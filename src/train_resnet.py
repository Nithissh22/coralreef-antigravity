"""
train_resnet.py — Deep learning training script for coral reef classification.

Supports:
    - ResNet50       (default, transfer learning)
    - EfficientNet-B0
    - ViT-B/16

CLI:
    python -m src.train_resnet
    python -m src.train_resnet --model efficientnet_b0 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import models
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger

logger = setup_logger("train_dl")


# ================================================================
#  Model factory
# ================================================================

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Return a torchvision model with its classifier head replaced."""
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if name == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=weights_arg)
        model.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.heads[0].in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all layers except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "vit_b_16":
        for param in model.heads.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ================================================================
#  Training loop
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += float(loss.item()) * images.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / total, correct / total


# ================================================================
#  Main
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep learning coral classifier")
    parser.add_argument("--model", type=str, default=None, help="resnet50 | efficientnet_b0 | vit_b_16")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]

    model_name: str   = args.model      or tcfg["model"]
    epochs: int       = args.epochs     or tcfg["epochs"]
    lr: float         = args.lr         or tcfg["learning_rate"]
    batch_size: int   = args.batch_size or tcfg["batch_size"]
    num_classes: int  = dcfg["num_classes"]
    seed: int         = dcfg.get("random_seed", 42)
    patience: int     = tcfg.get("early_stopping_patience", 7)
    unfreeze_at: int  = tcfg.get("unfreeze_after_epoch", 5)
    do_freeze: bool   = tcfg.get("freeze_backbone", True)

    set_seed(seed)
    device = get_device(tcfg.get("device", "auto"))
    logger.info("Device: %s | Model: %s | Epochs: %d", device, model_name, epochs)

    # Override batch_size in cfg for dataloader
    cfg["training"]["batch_size"] = batch_size
    train_loader, val_loader, _ = get_dataloaders(cfg)

    model = build_model(model_name, num_classes, pretrained=tcfg.get("pretrained", True))
    if do_freeze:
        freeze_backbone(model, model_name)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=lr, weight_decay=tcfg.get("weight_decay", 1e-4))

    scheduler_name = tcfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # --- Training ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    model_dir = ensure_dir(cfg["paths"]["models"])
    runs_dir = ensure_dir(cfg["paths"]["runs"])

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)

        # Unfreeze backbone after N epochs
        if do_freeze and epoch == unfreeze_at + 1:
            logger.info("🔓  Unfreezing backbone for full fine-tuning")
            unfreeze_all(model)
            optimizer = Adam(list(model.parameters()), lr=lr * 0.1,
                             weight_decay=float(tcfg.get("weight_decay", 1e-4)))

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 4),
        }
        history.append(record)
        logger.info(
            "  train_loss=%.4f  train_acc=%.4f  |  val_loss=%.4f  val_acc=%.4f",
            train_loss, train_acc, val_loss, val_acc,
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = model_dir / "best_model.pth"
            torch.save({
                "model_name": model_name,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
            logger.info("  💾  Saved best model (val_acc=%.4f) → %s", val_acc, ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("⏹  Early stopping triggered (patience=%d)", patience)
            break

    # Save history
    history_path = runs_dir / f"training_history_{model_name}.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training complete. History → %s", history_path)


if __name__ == "__main__":
    main()
