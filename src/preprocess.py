"""
preprocess.py — Underwater image enhancement pipeline.

Functions:
    - color_correct()       Gray-World white-balance
    - denoise()             Non-local means denoising
    - apply_clahe()         CLAHE on L-channel (LAB colour space)
    - enhance_underwater_image()  Full pipeline (above three combined)

CLI usage:
    python -m src.preprocess                       # uses config defaults
    python -m src.preprocess --input data/raw --output data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("preprocess")


# ================================================================
#  Individual enhancement steps
# ================================================================

def color_correct(image: np.ndarray) -> np.ndarray:
    """Apply Gray-World white-balance colour correction.

    Scales each channel so that the average colour of the scene is gray,
    which compensates for the blue / green colour-cast typical of
    underwater photographs.
    """
    result = image.astype(np.float32)
    avg_b, avg_g, avg_r = result.mean(axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def denoise(image: np.ndarray, h: int = 10, h_color: int = 10,
            template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Remove noise using OpenCV fast non-local means denoising (colour)."""
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h_color, template_window, search_window
    )


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE on the L-channel of the LAB colour space.

    This improves local contrast without amplifying noise in dark regions.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_chan = clahe.apply(l_chan)

    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ================================================================
#  Combined pipeline
# ================================================================

def enhance_underwater_image(
    image: np.ndarray,
    do_color_correct: bool = True,
    do_denoise: bool = True,
    do_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Run the full enhancement pipeline on a single BGR image."""
    if do_color_correct:
        image = color_correct(image)
    if do_denoise:
        image = denoise(image)
    if do_clahe:
        image = apply_clahe(image, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
    return image


# ================================================================
#  Batch processing (CLI)
# ================================================================

def batch_process(input_dir: Path, output_dir: Path, cfg: dict) -> None:
    """Enhance all images in *input_dir* (recursive) and mirror structure to *output_dir*."""
    pp = cfg.get("preprocessing", {})
    do_cc = pp.get("color_correction", True)
    do_dn = pp.get("denoise", True)
    do_cl = pp.get("clahe", True)
    clip = pp.get("clahe_clip_limit", 2.0)
    grid = tuple(pp.get("clahe_tile_grid_size", [8, 8]))

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]

    if not images:
        logger.warning("No images found in %s", input_dir)
        return

    logger.info("Found %d image(s) in %s", len(images), input_dir)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read %s — skipping", img_path)
            continue

        enhanced = enhance_underwater_image(
            img,
            do_color_correct=do_cc,
            do_denoise=do_dn,
            do_clahe=do_cl,
            clahe_clip=clip,
            clahe_grid=grid,
        )

        relative = img_path.relative_to(input_dir)
        out_path = output_dir / relative
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), enhanced)
        logger.info("✅  %s → %s", img_path.name, out_path)


# ================================================================
#  CLI entry-point
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Underwater image enhancement pipeline")
    parser.add_argument("--input", type=str, default=None, help="Input directory (default: from config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: from config)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(args.input) if args.input else Path(cfg["paths"]["raw_data"])
    output_dir = Path(args.output) if args.output else Path(cfg["paths"]["processed_data"])

    batch_process(input_dir, output_dir, cfg)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
