# Comparative Analysis: Decision Tree (J48) vs. Random Forest

This repository contains a local replication and validation study of the empirical findings published in the reputable benchmarking paper: **"Random forests and decision trees"** (Ali et al., 2012). 

The goal of this project is to implement the exact evaluation strategies outlined by the authors, match their specific hyperparameter baselines locally, and analyze any resulting variances.

---

## 📊 Evaluation Baseline & Target
The replication targets **Dataset 14 (Breast W / Breast Cancer Wisconsin)** from the paper, utilizing its specific data footprint (699 instances, 10 attributes) and testing methodologies.

### Experimental Configuration:
* **Validation Strategy:** 10-Fold Cross-Validation (`cv=10`)
* **Splitting Criterion:** Information Gain / Entropy (to match C4.5 framework)
* **Minimum Leaf Instances:** 2 (`min_samples_leaf=2`)
* **Ensemble Tree Count:** 10 Estimators (`n_estimators=10` to match historical Weka defaults)

---

## 📈 Performance Comparison Matrix

The local implementation yields numbers tightly aligned with the original 2012 published metrics:

| Model | Metric | Paper Baseline (2012 Weka J48 / RF) | Local Replication (Scikit-Learn) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Decision Tree** | Accuracy | 94.56% | **93.72%** | Validated |
| **Decision Tree** | Precision | 0.946 | **0.938** | Validated |
| **Decision Tree** | Recall | 0.946 | **0.925** | Validated |
| **Random Forest** | Accuracy | 96.13% | **96.36%** | Validated |
| **Random Forest** | Precision | 0.962 | **0.964** | Validated |
| **Random Forest** | Recall | 0.961 | **0.957** | Validated |

---

## 🔍 Analytical Breakdown of Variance

While the execution tightly replicates the paper's benchmark, minor variations are observed. This demonstrates expected behavior due to differences in software frameworks:

1. **Algorithmic Discrepancies (CART vs. C4.5):** The original paper utilized Weka's **J48** engine, which is a native implementation of the C4.5 algorithm capable of producing *multi-way splits*. Python's `scikit-learn` utilizes an optimized **CART** engine, which strictly enforces *binary tree splits*, slightly shifting the final decision boundaries.
2. **Pruning Methodologies:** J48 relies on a specialized post-pruning technique called *Subtree Raising* based on error confidence limits. Scikit-learn doesn't feature post-pruning natively, controlling tree growth through pre-pruning metrics (`min_samples_leaf`).
3. **Cross-Validation Fold Splits:** Minor random seed differences in row-shuffling across the 10 data folds introduce distinct internal data distributions per fold, shifting precision/recall values slightly.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3
* **Environment:** Google Colab
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`
