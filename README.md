# 🦅 DewHawks Internship — Machine Learning & AI Engineering Portfolio

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A consolidated repository tracking core engineering tasks, production-grade pipelines, and algorithmic studies completed during my engineering internship at **DewHawks**. 

This monorepo serves as an ongoing sandbox demonstrating multi-model implementations, performance benchmarking, and hyperparameter tuning across various machine learning domains.

---

## 🗂️ Project Dashboard

### [📨 01. Automated Multi-Model Spam Filter](./01_Spam_Classification)
* **Domain:** Natural Language Processing (NLP) & Text Classification
* **Core Stack:** Python, Scikit-Learn, TF-IDF Vectorizer
* **Summary:** An end-to-end classification pipeline comparing Multinomial Naive Bayes, SVM (RBF), and Decision Trees on the UCI SMS dataset. Focuses on tuning probability thresholds and kernel parameters to optimize precision and eliminate false positives.

### [🌲 02. Structural Decision Tree Benchmarking](./02_Decision_Tree_Benchmarking)
* **Domain:** Supervised Algorithmic Analysis
* **Core Stack:** Python, Scikit-Learn (J48/Decision Trees, Random Forest)
* **Summary:** An empirical benchmarking study evaluating variations in tree architecture. Analyzes decision boundary optimization, feature splitting criteria, and structural ensemble comparisons between singular trees and Random Forests.

---

## 🚀 Repository Setup & Usage

### 🛠️ Global Installation
To install the dependencies for all projects across this repository, clone the workspace and run the requirement installation inside the target task directory:

```bash
# Clone the repository
git clone [https://github.com/MajidFQ/DewHawks-Internship-Tasks.git](https://github.com/MajidFQ/DewHawks-Internship-Tasks.git)
cd DewHawks-Internship-Tasks

# Navigate to a specific task to run its pipeline
cd 01_Spam_Classification
pip install -r requirements.txt
