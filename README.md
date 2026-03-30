# MAL-ton-iot-ids

A machine learning-based **Intrusion Detection System (IDS)** for IoT network traffic, built on the [TON-IoT dataset](https://research.unsw.edu.au/projects/toniot-datasets) from UNSW. Multiple supervised and unsupervised models are trained, evaluated, and compared to detect malicious network activity.

---

## 📋 Overview

This project explores and benchmarks several ML approaches for detecting network intrusions in IoT environments. Both **supervised classifiers** and **unsupervised anomaly detection** methods are implemented. Models are evaluated with the **F2-score**, which prioritises recall — minimising missed attacks is more important than avoiding false alarms in a security context.

Resource consumption (CPU, memory, energy) is also measured to compare model efficiency.

---

## 📁 Repository Structure

```
sys-src/                          # Source code and Jupyter notebooks
├── ton_iot_pipeline.py           # Core pipeline: data loading, preprocessing, training, evaluation
├── ton_iot_utils.py              # Utilities: custom transformers, resource & energy monitor
├── preprocessing.py              # Preprocessing classes for Isolation Forest & CatBoost
├── ton_iot_train_svm.ipynb       # Train SVM models (SVC, LinearSVC, SGD, OCSVM)
├── ton_iot_inference.ipynb       # Run inference with saved models
├── ton_iot_compare_results.ipynb # Compare performance and efficiency across all models
└── balzer/
    ├── eda.ipynb                     # Exploratory Data Analysis
    ├── catboost.ipynb                # CatBoost classifier training
    ├── iforest_raw.ipynb             # Isolation Forest on raw features
    ├── iforest_optimized.ipynb       # Isolation Forest on engineered features
    ├── iforest_tuned_raw.ipynb       # Hyperparameter-tuned Isolation Forest (raw)
    └── iforest_tuned_optimized.ipynb # Hyperparameter-tuned Isolation Forest (optimized)

sys-doc/                          # Project documentation
├── projektarbeit.pdf / .tex      # Full project report
├── abschlusspraesentation.pdf    # Final presentation slides
├── quellen.bib                   # Bibliography
└── requirements.txt              # Python dependency list
```

---

## 🗃️ Dataset

**TON-IoT `train_test_network`** — a network traffic dataset from UNSW Sydney covering normal and attack traffic across various IoT scenarios.

| Detail | Value |
|--------|-------|
| Source | [UNSW TON-IoT Datasets](https://research.unsw.edu.au/projects/toniot-datasets) |
| Expected path | `sys-src/data/train_test_network.csv` |
| Labels | Binary: `0` = normal, `1` = attack |
| Attack types | Multi-class `type` column used for stratification |
| Data split | 70% train / 15% validation / 15% test (stratified by attack type) |

---

## 🤖 Models

| Model | Type | Notebook |
|-------|------|----------|
| SVC | Supervised | `ton_iot_train_svm.ipynb` |
| LinearSVC | Supervised | `ton_iot_train_svm.ipynb` |
| SGDClassifier | Supervised | `ton_iot_train_svm.ipynb` |
| One-Class SVM (OCSVM) | Unsupervised (anomaly detection) | `ton_iot_train_svm.ipynb` |
| Isolation Forest | Unsupervised (anomaly detection) | `balzer/iforest_*.ipynb` |
| CatBoost | Supervised (gradient boosting) | `balzer/catboost.ipynb` |

---

## 📊 Evaluation

- **Primary metric:** F2-score (favours recall over precision)
- **Additional metrics:** Accuracy, Precision, Recall, Confusion Matrix
- **Resource tracking:** CPU usage, RAM, wall-clock time, and energy consumption via [CodeCarbon](https://codecarbon.io/)

Use `ton_iot_compare_results.ipynb` to load saved inference results and compare all models side by side.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r sys-doc/requirements.txt
```

### 2. Download the dataset

Download `train_test_network.csv` from the [UNSW TON-IoT page](https://research.unsw.edu.au/projects/toniot-datasets) and place it at:

```
sys-src/data/train_test_network.csv
```

### 3. Run the notebooks

| Step | Notebook |
|------|----------|
| Exploratory Data Analysis | `sys-src/balzer/eda.ipynb` |
| Train SVM / OCSVM models | `sys-src/ton_iot_train_svm.ipynb` |
| Train Isolation Forest | `sys-src/balzer/iforest_*.ipynb` |
| Train CatBoost | `sys-src/balzer/catboost.ipynb` |
| Run inference | `sys-src/ton_iot_inference.ipynb` |
| Compare all results | `sys-src/ton_iot_compare_results.ipynb` |

### Pre-trained models

Pre-trained models and pipelines are available for download:
👉 [Models + Pipelines (Dropbox)](https://www.dropbox.com/scl/fi/34yrdv4yrbnfcc9ek6gxj/ton-iot-models.zip?rlkey=20nsxo93s39ozuopejxibrfko&dl=0)

---

## 👥 Authors

| Author | Contributions |
|--------|--------------|
| **Kevin Paulus** | Pipeline, SVM models, One-Class SVM, utility classes |
| **Raphael Balzer** | Isolation Forest, CatBoost, preprocessing, EDA |

*GitHub Copilot and Claude Sonnet were used to assist with code and refactoring.*
