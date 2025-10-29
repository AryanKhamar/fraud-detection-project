# 💳 Fraud Transaction Detection System

This project builds a **machine learning system** that classifies whether a financial transaction is **fraudulent or legitimate** using historical transaction patterns.  
The dataset simulates real-world transaction behavior and includes multiple types of fraud events.

---

## Run
1. Install: `pip install -r requirements.txt`
2. Train: `python main.py`
3. Predict: `python predict_new.py`
4. Analysis: `jupyter notebook notebooks/exploratory_analysis.ipynb`

## 📘 Project Overview

The objective of this project is to:

> **Build a system that can classify if a transaction is fraudulent or not.**

It involves:
- Loading and preprocessing multiple daily transaction files (`.pkl` format)
- Engineering behavioral and temporal features
- Training baseline (Logistic Regression) and advanced (XGBoost) models
- Evaluating models using fraud-specific metrics
- Predicting fraud probability for new transactions

---

## 🧩 Fraud Scenarios in Dataset

The dataset simulates 3 types of fraud:

| Scenario | Description | Detection Method |
|-----------|--------------|------------------|
| **1️⃣ High Amount Fraud** | Transactions where `TX_AMOUNT > 220` are fraudulent | Amount-based threshold |
| **2️⃣ Terminal Fraud** | 2 terminals chosen daily; all their transactions for the next 28 days are fraud | Terminal rolling fraud rate |
| **3️⃣ Customer Fraud** | 3 customers chosen daily; 1/3 of their transactions in the next 14 days are 5× higher in amount and marked as fraud | Customer spending deviation |

---

## 🚀 Run the Project

1️⃣ **Clone or Download this Repository**
```bash
git clone https://github.com/AryanKhamar/fraud-detection-project.git
cd fraud-detection-project
