# 🐠 Coral Reef Health Classification System

AI-powered image classification to detect whether coral reef images are **Healthy**, **Bleached**, or **Diseased** — with Grad-CAM explainability for marine biologists and conservationists.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Project Structure

```
coralreef-antigravity/
├── configs/
│   └── default.yaml          # Central configuration
├── data/
│   ├── raw/                   # Original images (healthy/ bleached/ diseased/)
│   └── processed/             # Enhanced images after preprocessing
├── models/                    # Saved model checkpoints
├── runs/                      # Training logs & Grad-CAM outputs
│   └── gradcam/
├── src/
│   ├── __init__.py
│   ├── utils.py               # Shared helpers (seed, device, logging)
│   ├── preprocess.py          # Underwater image enhancement pipeline
│   ├── dataset.py             # PyTorch Dataset & DataLoaders
│   ├── train_resnet.py        # Deep learning training (ResNet / EfficientNet / ViT)
│   ├── train_ml.py            # Traditional ML (SVM, Random Forest, XGBoost)
│   ├── eval.py                # Evaluation metrics & confusion matrix
│   └── explain.py             # Grad-CAM explainability
├── app.py                     # Streamlit deployment dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Place coral images into class sub-folders:

```
data/raw/
  ├── healthy/
  ├── bleached/
  └── diseased/
```

You can use datasets from [CoralNet](https://coralnet.ucsd.edu/), [NOAA](https://www.noaa.gov/), or [Kaggle coral datasets](https://www.kaggle.com/search?q=coral+reef).

### 3. Preprocess images

Enhance underwater images with color correction, denoising, and CLAHE:

```bash
python -m src.preprocess
```

Enhanced images are saved to `data/processed/`.

### 4. Train deep learning model

```bash
# ResNet50 (default)
python -m src.train_resnet

# EfficientNet-B0
python -m src.train_resnet --model efficientnet_b0

# Vision Transformer
python -m src.train_resnet --model vit_b_16
```

### 5. Train traditional ML baselines

```bash
python -m src.train_ml
```

### 6. Evaluate models

```bash
# Evaluate deep learning model
python -m src.eval --model-type dl --checkpoint models/best_model.pth

# Evaluate traditional ML model
python -m src.eval --model-type ml --checkpoint models/random_forest.joblib
```

### 7. Generate Grad-CAM heatmaps

```bash
# Single image
python -m src.explain --image path/to/coral.jpg

# Batch (entire test set)
python -m src.explain --batch
```

### 8. Launch Streamlit dashboard

```bash
streamlit run app.py
```

Upload a coral image → the app returns the predicted health status, confidence scores, and a Grad-CAM heatmap overlay.

---

## 🧠 Models

| Model | Type | Features |
|-------|------|----------|
| ResNet50 | Deep Learning | Transfer learning, fine-tuned classifier |
| EfficientNet-B0 | Deep Learning | Efficient compound scaling |
| ViT-B/16 | Deep Learning | Vision Transformer with patch embeddings |
| SVM (RBF) | Traditional ML | Color histogram + Haralick + HOG |
| Random Forest | Traditional ML | Color histogram + Haralick + HOG |
| XGBoost | Traditional ML | Color histogram + Haralick + HOG |

---

## 🔍 Explainability

Grad-CAM heatmaps highlight the coral regions the model focuses on, helping marine biologists understand *why* a classification was made.

---

## 📊 Evaluation Metrics

- **Accuracy** — overall correctness
- **Precision / Recall / F1-score** — per-class performance
- **Confusion matrix** — saved as PNG to `runs/`

---

## 🛠️ Configuration

All hyperparameters are centralized in `configs/default.yaml` — image size, batch size, learning rate, augmentation, model selection, and more.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
