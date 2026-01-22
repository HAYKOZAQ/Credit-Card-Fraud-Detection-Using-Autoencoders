# 🛡️ Credit Card Fraud Detection using Autoencoders

## 📌 Project Overview

This project implements a robust, enterprise-grade **Fraud Detection System** using various **Deep Learning Autoencoder architectures**. Unlike traditional supervised learning that relies heavily on labeled fraud data, this system focuses on **Unsupervised Anomaly Detection**, learning the latent representation of "normal" transactions to identify deviations (fraud) without needing massive datasets of flagged fraud.

The project goes beyond simple modeling to include **MLOps best practices**, **Concept Drift Simulation**, **Explainable AI (SHAP)**, and a **Real-time Dashboard**.

---

## 🚀 Key Features

### 🧠 Deep Learning Architectures

We implemented and compared multiple Autoencoder variants:

* **Standard Autoencoder:** The baseline for reconstruction error-based detection.
* **Variational Autoencoder (VAE):** Learns a probabilistic latent space for better generalization.
* **Denoising Autoencoder:** Trained to remove noise, making the model robust to corrupted inputs.
* **Contrastive Autoencoder:** Optimizes the latent space separation between normal and anomalous patterns (Best Unsupervised Performer).
* **Graph Autoencoder:** Incorporates relationship data (graph topology) between entities.
* **LSTM Autoencoder:** Handles sequential transaction data to detect temporal anomalies.

### 🛠️ Advanced Capabilities

* **Concept Drift Simulation:** A dedicated engine (`src/simulation.py`) to simulate data distribution shifts over months, proving the need for adaptive retraining.
* **Active Learning Loop:** A simulation of human-in-the-loop feedback to iteratively improve the model with minimal labeling budget.
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to explain *why* a specific transaction was flagged (e.g., "high amount" + "unusual location").
* **Federated Learning:** A simulation of privacy-preserving training across multiple banks without sharing raw data.
* **Adversarial Robustness:** Tested model performance against adversarial attacks.

### 💻 Interfaces

* **Streamlit Dashboard:** A "Research Lab" and "Fraud Inspector" UI for interacting with the models.
* **Gradio App:** A production-style interface for real-time inference.

---

## 📊 Results & Analysis

### 1. Comparative Performance (Research Metrics)

We benchmarked our Unsupervised Autoencoders against a Supervised XGBoost baseline. While the Supervised model (XGBoost) naturally performs best when abundant labels are available, our **Contrastive Autoencoder** demonstrated the best performance among unsupervised methods, showing strong ability to detect fraud without reliance on labels.

| Model                             | AUPRC     | AUROC     | Detection Cost ($) | Precision | Recall |
| --------------------------------- | --------- | --------- | ------------------ | --------- | ------ |
| **XGBoost (Supervised Baseline)** | **0.935** | **0.995** | $21,130            | 0.887     | 0.893  |
| **Contrastive AE**                | **0.326** | 0.800     | $118,090           | 0.521     | 0.371  |
| **Variational AE (VAE)**          | 0.272     | **0.818** | $120,200           | 0.427     | 0.376  |
| **Graph AE**                      | 0.271     | 0.776     | $127,400           | 0.416     | 0.332  |
| **Standard AE**                   | 0.181     | 0.773     | $130,570           | 0.338     | 0.332  |

* **Insight:** The **Contrastive Autoencoder** significantly outperformed the Standard AE (AUPRC 0.32 vs 0.18), proving that structuring the latent space is critical for anomaly detection.
* **Business Impact:** The Contrastive model saved ~$12,000 more in fraud losses compared to the Standard AE.

### 2. Concept Drift Simulation

We simulated a 4-month timeline where transaction patterns shifted.

* **Static Model:** Performance collapsed in Month 1 (AUPRC dropped to **0.005**).
* **Adaptive Model:** Retraining allowed the model to recover and maintain performance (AUPRC **~0.20-0.30**).
* **Conclusion:** Static models fail in production; continuous retraining is mandatory.

---

## 📂 Project Structure

```text
├── main.py                  # Entry point for Research, Simulation, and Advanced suites
├── dashboard.py             # Streamlit Interactive Dashboard
├── app.py                   # Gradio Production Interface
├── ablation_study.py        # Script for component analysis
├── requirements.txt         # Dependencies
├── src/
│   ├── model.py             # PyTorch definitions for all Autoencoders
│   ├── data_loader.py       # Data preprocessing, sequence creation, loaders
│   ├── train.py             # Training loops
│   ├── evaluate.py          # Evaluation logic (errors, plots)
│   ├── metrics.py           # FraudEvaluator (AUPRC, Cost, Impact)
│   ├── simulation.py        # DriftSimulator class
│   ├── active_learning.py   # Active Learning logic
│   ├── federated.py         # Federated Learning simulation
│   └── explain.py           # SHAP explanation wrapper
└── results/                 # CSV logs of all experiments
```

---

## ⚙️ How to Run

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Main Research Suite

To train models and generate the comparison metrics:

```bash
python main.py --mode research
```

### 3. Run Advanced Simulations

To run the Concept Drift, Active Learning, and Robustness suites:

```bash
python main.py --mode advanced
```

### 4. Launch the Dashboard

To explore the models visually:

```bash
streamlit run dashboard.py
```

*(Access at <http://localhost:8501>)*

### 5. Launch Production App

```bash
python app.py
```

---

## 🔎 Data & Preprocessing

The system expects a transaction dataset with the following features (engineered in `src/data_loader.py`):

* **Inputs:** `category`, `job`, `city_pop`, `age`, `hour`, `distance_km` (Haversine), `amt_log` (Log Amount).
* **Preprocessing:** Log scaling for amounts, Label Encoding for categoricals, Standard Scaling for all inputs.
* **Sequences:** For LSTM/RNN models, transactions are grouped by `cust_id` or `merchant` to form temporal sequences.

---

## 📈 Future Work

* **Graph Neural Networks (GNN):** Fully exploit the transaction graph structure beyond the simple Graph Convolution implemented.
* **Real-time Pipeline:** Connect the system to a real message queue (Kafka) for streaming inference.
* **Transformer Models:** Replace LSTM with Transformer-based anomaly detection for better long-range dependency capture.
