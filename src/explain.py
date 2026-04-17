"""
explain.py — Grad-CAM explainability for coral reef classification.

Generates heatmap overlays showing which coral regions the model focuses on.

CLI:
    python -m src.explain --image path/to/coral.jpg
    python -m src.explain --batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.dataset import get_transforms
from src.train_resnet import build_model
from src.utils import ensure_dir, get_device, load_config, setup_logger

logger = setup_logger("explain")

# Map config strings to pytorch_grad_cam classes
CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
}


# ================================================================
#  Target layer auto-detection
# ================================================================

def get_target_layer(model: torch.nn.Module, model_name: str):
    """Return the last convolutional layer for Grad-CAM."""
    if model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    elif model_name == "vit_b_16":
        return [model.encoder.layers[-1].ln_1]
    else:
        raise ValueError(f"Cannot auto-detect target layer for: {model_name}")


# ================================================================
#  Core Grad-CAM function
# ================================================================

def generate_gradcam(
    image_path: str | Path,
    checkpoint_path: str | Path,
    cfg: dict | None = None,
    method: str = "gradcam",
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generate a Grad-CAM heatmap for a single image.

    Returns:
        (original_rgb, cam_overlay, predicted_class, confidence)
    """
    if cfg is None:
        cfg = load_config()

    device = get_device(cfg["training"].get("device", "auto"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", cfg["training"]["model"])
    num_classes = ckpt.get("num_classes", cfg["dataset"]["num_classes"])

    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Prepare image
    image_size = cfg["dataset"].get("image_size", 224)
    tf = get_transforms(image_size, mode="val")
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = tf(pil_img).unsqueeze(0).to(device)

    # Original image normalised to [0, 1] for overlay
    rgb_img = np.array(pil_img.resize((image_size, image_size))).astype(np.float32) / 255.0

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Grad-CAM
    target_layers = get_target_layer(model, model_name)
    cam_class = CAM_METHODS[method] if method in CAM_METHODS else GradCAM

    targets = [ClassifierOutputTarget(target_class if target_class is not None else pred_class)]

    with cam_class(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return rgb_img, cam_overlay, pred_class, confidence


# ================================================================
#  CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM explainability for coral classifier")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--batch", action="store_true", help="Process all test images")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (default: models/best_model.pth)")
    parser.add_argument("--method", type=str, default=None,
                        choices=list(CAM_METHODS.keys()))
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    xcfg = cfg.get("explainability", {})
    method = args.method or xcfg.get("method", "gradcam")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["models"]) / "best_model.pth"
    output_dir = ensure_dir(xcfg.get("output_dir", "runs/gradcam"))
    class_names = cfg["dataset"]["classes"]

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    if args.image:
        # Single image mode
        img_path = Path(args.image)
        _, cam_overlay, pred, conf = generate_gradcam(
            img_path, ckpt_path, cfg, method=method
        )
        out_name = f"{img_path.stem}_gradcam.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
        logger.info(
            "Prediction: %s (%.2f%%) — heatmap saved → %s",
            class_names[pred], conf * 100, out_path,
        )

    elif args.batch:
        # Batch mode — iterate over processed data
        data_dir = Path(cfg["paths"]["processed_data"])
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [p for p in data_dir.rglob("*") if p.suffix.lower() in extensions]
        logger.info("Generating Grad-CAM for %d images …", len(images))

        for img_path in images:
            _, cam_overlay, pred, conf = generate_gradcam(
                img_path, ckpt_path, cfg, method=method
            )
            rel = img_path.relative_to(data_dir)
            out_path = output_dir / rel.parent / f"{img_path.stem}_gradcam.png"
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        logger.info("Batch Grad-CAM complete. Outputs → %s", output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
