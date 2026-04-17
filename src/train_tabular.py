"""
train_tabular.py — Train a tabular machine learning model on a CSV dataset.

This script demonstrates training a model (Random Forest / XGBoost)
using a CSV dataset with numerical and categorical features related
to coral reef health. If the dataset does not exist, it will generate
a synthetic one for demonstration purposes.

CLI:
    python -m src.train_tabular
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_tabular")

def generate_synthetic_data(csv_path: Path, num_samples: int = 1000) -> pd.DataFrame:
    """Generate a synthetic CSV dataset for coral health if none exists."""
    logger.info("Generating synthetic dataset with %d samples...", num_samples)
    np.random.seed(42)
    
    # Generate features
    temperatures = np.random.normal(29.0, 2.0, num_samples)
    depths = np.random.uniform(2.0, 40.0, num_samples)
    salinity = np.random.normal(35.0, 1.5, num_samples)
    pollution_index = np.random.uniform(0.0, 10.0, num_samples)
    
    # Determine labels based on simple logical rules to simulate a real scenario
    labels = []
    for t, d, s, p in zip(temperatures, depths, salinity, pollution_index):
        if t > 31.0 or (t > 30.0 and p > 7.0):
            labels.append("bleached")
        elif p > 8.0 or (s < 32.0 and p > 5.0):
            labels.append("diseased")
        else:
            labels.append("healthy")
            
    # Introduce some noise to make the ML task realistic
    noise_indices = np.random.choice(num_samples, int(num_samples * 0.1), replace=False)
    classes = ["healthy", "bleached", "diseased"]
    for idx in noise_indices:
        labels[idx] = np.random.choice(classes)
        
    df = pd.DataFrame({
        "temperature_c": temperatures,
        "depth_m": depths,
        "salinity_ppt": salinity,
        "pollution_index": pollution_index,
        "health_status": labels
    })
    
    # Save to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved synthetic dataset to %s", csv_path)
    return df

def main():
    parser = argparse.ArgumentParser(description="Train tabular ML model on CSV data")
    parser.add_argument("--data", type=str, default="data/tabular/coral_health.csv", help="Path to CSV dataset")
    parser.add_argument("--model-out", type=str, default="models/tabular_rf.joblib", help="Path to save the trained model")
    args = parser.parse_args()
    
    csv_path = Path(args.data)
    
    # Generate data if it doesn't exist
    if not csv_path.exists():
        df = generate_synthetic_data(csv_path)
    else:
        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
        
    if df.empty:
        logger.error("Dataset is empty.")
        return
        
    # Preprocessing
    target_col = "health_status"
    if target_col not in df.columns:
        logger.error("Target column '%s' not found in dataset.", target_col)
        return
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training Random Forest Classifier on %d samples...", len(X_train))
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Testing Accuracy: %.2f%%", acc * 100)
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info("\nClassification Report:\n%s", report)
    
    # Save artifacts
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": X.columns.tolist()
    }
    joblib.dump(artifact, out_path)
    logger.info("💾 Saved trained model and preprocessors to %s", out_path)

if __name__ == "__main__":
    main()
