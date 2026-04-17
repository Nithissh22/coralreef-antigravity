"""
train_ml.py — Traditional ML baselines for coral reef classification.

Models:  SVM (RBF), Random Forest, XGBoost
Features: Colour histogram, Haralick texture (GLCM), HOG

CLI:
    python -m src.train_ml
    python -m src.train_ml --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils import ensure_dir, load_config, set_seed, setup_logger

logger = setup_logger("train_ml")


# ================================================================
#  Feature extraction
# ================================================================

def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute a normalised colour histogram (BGR, concatenated)."""
    features = []
    for ch in range(3):
        hist = cv2.calcHist([image], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    return np.concatenate(features)


def extract_haralick(image: np.ndarray) -> np.ndarray:
    """Compute Haralick texture features from the gray-scale GLCM."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Quantise to fewer levels to keep GLCM manageable
    gray = (gray // 4).astype(np.uint8)  # 64 levels
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = np.hstack([graycoprops(glcm, p).ravel() for p in props])
    return feats


def extract_hog(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """Compute HOG descriptor on a resized gray-scale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size))
    hog = cv2.HOGDescriptor(
        _winSize=(image_size, image_size),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(gray).flatten()


def extract_features(image: np.ndarray, feature_list: list[str]) -> np.ndarray:
    """Extract and concatenate requested feature vectors."""
    parts = []
    if "color_histogram" in feature_list:
        parts.append(extract_color_histogram(image))
    if "haralick" in feature_list:
        parts.append(extract_haralick(image))
    if "hog" in feature_list:
        parts.append(extract_hog(image))
    return np.concatenate(parts)


# ================================================================
#  Data loading (flat images → feature matrix)
# ================================================================

def load_features(
    data_dir: Path,
    class_names: list[str],
    feature_list: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read images from class sub-folders, extract features, return X, y."""
    X_list, y_list = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for idx, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            logger.warning("Missing class directory: %s", cls_dir)
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feats = extract_features(img, feature_list)
            X_list.append(feats)
            y_list.append(idx)

    logger.info("Extracted features for %d images", len(X_list))
    return np.array(X_list), np.array(y_list)


# ================================================================
#  Model training
# ================================================================

def build_ml_models(cfg: dict) -> dict:
    """Instantiate ML models based on config."""
    ml_cfg = cfg.get("traditional_ml", {})
    models_to_train: dict = {}

    model_names = ml_cfg.get("models", ["svm", "random_forest", "xgboost"])

    if "svm" in model_names:
        svm_cfg = ml_cfg.get("svm", {})
        models_to_train["svm"] = SVC(
            kernel=svm_cfg.get("kernel", "rbf"),
            C=svm_cfg.get("C", 10.0),
            probability=True,
        )

    if "random_forest" in model_names:
        rf_cfg = ml_cfg.get("random_forest", {})
        models_to_train["random_forest"] = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth"),
            random_state=cfg["dataset"].get("random_seed", 42),
        )

    if "xgboost" in model_names:
        xgb_cfg = ml_cfg.get("xgboost", {})
        models_to_train["xgboost"] = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            random_state=cfg["dataset"].get("random_seed", 42),
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

    return models_to_train


# ================================================================
#  Main
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional ML coral classifiers")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["dataset"].get("random_seed", 42))

    class_names = cfg["dataset"]["classes"]
    feature_list = cfg["traditional_ml"].get("features", ["color_histogram", "haralick", "hog"])
    data_dir = Path(cfg["paths"]["processed_data"])

    logger.info("Loading features from %s …", data_dir)
    X, y = load_features(data_dir, class_names, feature_list)

    if len(X) == 0:
        logger.error("No images found — cannot train. Check data directory.")
        return

    # Train / test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,
        random_state=cfg["dataset"].get("random_seed", 42),
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_dir = ensure_dir(cfg["paths"]["models"])
    joblib.dump(scaler, model_dir / "scaler.joblib")

    ml_models = build_ml_models(cfg)
    for name, model in ml_models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names)
        logger.info("\n%s classification report:\n%s", name, report)

        model_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info("💾  Saved %s → %s", name, model_path)


if __name__ == "__main__":
    main()
