# 📨 Automated Multi-Model Spam Filter & Hyperparameter Optimization Benchmark

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-ready machine learning pipeline evaluating NLP-based email and SMS spam filtering using **Multinomial Naive Bayes**, **Support Vector Machines (SVM)**, and **Decision Tree Classifiers**. 

This repository serves as an empirical study on how adjusting classification thresholds, structural tree depths, and kernel radiuses drives models between underfitting, optimal generalization, and overfitting states.

---

## 📌 Project Overview & Architecture

Modern email systems must balance filter strength against user disruption. This project builds a TF-IDF text vectorization pipeline and trains three distinctly structured classifiers on the real-world **UCI SMS Spam Collection dataset** (5,574 messages). 

### Key Research Objectives:
* **Precision Optimization:** Mitigating destructive False Positives (routing critical alerts to Spam).
* **Boundary Tuning:** Mapping the behavior of decision boundaries across varying mathematical parameters.



---

## 📊 Deep-Dive Performance Matrix

The pipeline dynamically evaluates all model variations across five core metrics. 

| Model Family | Hyperparameter Configuration | Accuracy | Precision | Recall | F1-Score | Confusion Matrix `[TN, FP \| FN, TP]` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Probability Threshold: `0.3` | **0.9839** | 0.9580 | **0.9195** | **0.9384** | `960, 12 \| 12, 137` |
| **Naive Bayes** | Probability Threshold: `0.5` | 0.9794 | **1.0000** | 0.8456 | 0.9164 | `966, 0 \| 23, 126` |
| **Naive Bayes** | Probability Threshold: `0.75` | 0.9453 | **1.0000** | 0.5906 | 0.7426 | `966, 0 \| 61, 88` |
| **Decision Tree**| Max Structural Depth: `25` | 0.9650 | 0.9167 | 0.8121 | 0.8612 | `955, 11 \| 28, 121` |
| **Decision Tree**| Max Structural Depth: `50` | 0.9695 | 0.9197 | 0.8456 | 0.8811 | `955, 11 \| 23, 126` |
| **Decision Tree**| Max Structural Depth: `75` | 0.9695 | 0.9197 | 0.8456 | 0.8811 | `955, 11 \| 23, 126` |
| **SVM (RBF)** | Kernel Gamma: `0.01` | 0.8664 | 0.0000 | 0.0000 | 0.0000 | `966, 0 \| 149, 0` |
| **SVM (RBF)** | Kernel Gamma: `0.1` | 0.9561 | **1.0000** | 0.6711 | 0.8032 | `966, 0 \| 49, 100` |
| **SVM (RBF)** | Kernel Gamma: `0.5` | 0.9776 | **1.0000** | 0.8322 | 0.9084 | `966, 0 \| 25, 124` |

---

## 🧠 Engineering Insights & Hyperparameter Analysis

### 1. The Operational Trade-Off: Precision vs. Recall
In production environments, **False Positives are highly unacceptable**. If an urgent university deadline or a job notification is marked as spam, the system fails the user. 
* Setting the **Naive Bayes Threshold to 0.5 or 0.75** achieves a flawless **1.0000 Precision** (0 safe emails lost). 
* The trade-off is a drop in **Recall**, meaning the filter becomes conservative and lets tricky spam slip into the main inbox.

### 2. Decision Tree Depth Saturation
Increasing `max_depth` from `25` to `50` yields solid optimization gains. However, scaling from `50` to `75` shows **identical performance metrics**. This indicates structural saturation; the feature space is cleanly separated, and further depth introduces zero computational advantage.

### 3. Support Vector Machine Kernel Underfitting
At `Gamma = 0.01`, the RBF kernel's radius of influence is too broad. The model suffers from severe underfitting and defaults completely to predicting the majority class ("Not Spam" / Ham), resulting in a **0.0000 F1-Score**. Tuning Gamma upward to `0.5` constructs a crisp high-dimensional boundary, matching top-tier performance.

---

## 🚀 Quick Start & Replication

### 🛠️ Environment Setup
Clone the repository and install the standard scientific Python dependencies:
```bash
git clone [https://github.com/MajidFQ/MultiModel-Spam-Classification.git](https://github.com/MajidFQ/MultiModel-Spam-Classification.git)
cd MultiModel-Spam-Classification
pip install -r requirements.txt
