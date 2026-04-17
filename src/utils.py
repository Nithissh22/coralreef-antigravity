"""
utils.py — Shared helpers for the Coral Reef Health Classification project.

Provides:
    - set_seed()       Reproducible experiments
    - get_device()     Auto-detect CPU / CUDA
    - load_config()    Read YAML configuration
    - setup_logger()   Consistent logging across modules
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
def get_device(preference: str = "auto") -> torch.device:
    """Return a torch device based on *preference* ('auto', 'cpu', 'cuda')."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config file. Falls back to ``configs/default.yaml``."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def setup_logger(name: str = "coralreef", level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_class_names(cfg: dict | None = None) -> list[str]:
    """Return the list of class names from config (default: loaded from file)."""
    if cfg is None:
        cfg = load_config()
    return cfg["dataset"]["classes"]
