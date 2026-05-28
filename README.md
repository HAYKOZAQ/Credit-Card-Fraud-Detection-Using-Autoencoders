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

We benchmarked our Unsupervised Autoencoders against a Supervised XGBoost baseline. By engineering 25 features (leveraging location distance, categories, amount distribution statistics, and temporal patterns), the **Graph Autoencoder** achieves the best overall performance, surpassing even the supervised baseline by exploiting node relationship graphs.

| Model                             | AUPRC     | AUROC     | Detection Cost ($) | Precision | Recall |
| --------------------------------- | --------- | --------- | ------------------ | --------- | ------ |
| **Graph AE**                      | **0.982** | **1.000** | **$3,480**         | **0.840** | **0.999**|
| **XGBoost (Supervised Baseline)** | 0.873     | 0.993     | $15,160            | 0.544     | 0.906  |
| **Denoising AE**                  | 0.330     | 0.899     | $78,990            | 0.282     | 0.746  |
| **Variational AE (VAE)**          | 0.330     | 0.900     | $88,890            | 0.267     | 0.691  |
| **Standard AE**                   | 0.262     | 0.875     | $106,890           | 0.237     | 0.590  |
| **Contrastive AE**                | 0.232     | 0.868     | $110,590           | 0.231     | 0.569  |

* **Insight:** Incorporating graph topology (Graph AE) leads to a massive boost in performance, achieving **99.9% recall** and **84.0% precision** while using unsupervised learning.
* **Business Impact:** Graph AE reduces the total fraud detection loss from $127,400 to **$3,480** (a **97% cost reduction**), saving **$11,680** more than the supervised XGBoost baseline.

### 2. Concept Drift Simulation

We simulated a 4-month timeline where transaction patterns shifted.

* **Static Model:** Performance collapsed in Month 1 (AUPRC dropped to **0.005**).
* **Adaptive Model:** Retraining allowed the model to recover and maintain performance (AUPRC **~0.20-0.30**).
* **Conclusion:** Static models fail in production; continuous retraining is mandatory.

---

## 📂 Project Structure

```text
├── main.py                  # Entry point for Research, Simulation, and Advanced suites
├── dashboard.py             # Streamlit Interactive Dashboard (Expanded with tabs & SHAP)
├── app.py                   # Gradio Production Interface (With dynamic architecture selection)
├── api.py                   # Production FastAPI REST service
├── ablation_study.py        # Script for component analysis
├── Dockerfile               # Production Docker container definition
├── docker-compose.yml       # Full stack deployment (API, Dashboard, Prometheus, Grafana)
├── requirements.txt         # App/API dependencies
├── src/
│   ├── core/
│   │   ├── config.py        # Centralized configurations & feature lists
│   │   ├── data_loader.py   # Data preprocessing, sequence building, statistics loader
│   │   ├── model.py         # PyTorch definitions for 8 Autoencoder architectures
│   │   └── metrics.py       # FraudEvaluator calculations (AUPRC, Cost, Recall)
│   ├── training/
│   │   ├── train.py         # Baseline and adversarial training loops
│   │   ├── active_learning.py # Active Learning loop simulator
│   │   ├── federated.py     # Privacy-preserving federated simulation
│   │   └── gan_trainer.py   # GAN-based synthetic sample generation
│   ├── evaluation/
│   │   ├── evaluate.py      # Validation runs & custom SHAP bar aggregators
│   │   ├── explain.py       # Optimized chunked SHAP explainer
│   │   ├── robustness.py    # FGSM evasion attack evaluators
│   │   ├── simulation.py    # Drift timeline simulator
│   │   ├── baselines.py     # Supervised baseline wrappers
│   │   ├── hybrid.py        # Hybrid XGB + AE classifier
│   │   └── analysis_utils.py # Counterfactual and PSI drift calculators
│   └── monitoring/
│       └── monitor.py       # Production pipeline metric exporter
└── results/                 # CSV logs of all experiments
```

---

## 🛠️ Recent Upgrades & Refactoring

The system was updated to migrate from an experimental state to a fully robust production system:

1. **FastAPI Stats Loading Bug Fixed**: Corrected the statistics lookup file in [api.py](file:///c:/Users/zakar/Desktop/DATA/Credit-Card-Fraud-Detection-Using-Autoencoders/api.py) to point to the correct `stats.pkl` instead of a non-existent file, enabling correct category/merchant engineering.
2. **Dynamic Gradio Model Architectures**: Added an interactive selector in [app.py](file:///c:/Users/zakar/Desktop/DATA/Credit-Card-Fraud-Detection-Using-Autoencoders/app.py) for picking different model types (Standard, VAE, Graph AE, etc.), dynamically loading and caching their pre-trained weights, and setting separate reconstruction thresholds.
3. **Graph AE Memory Bottleneck Fixed**: Fixed an O(N^2) memory spike where Kernel SHAP's large perturbation batches caused PyTorch to attempt a 44 GB memory allocation. Re-implemented calculation in chunks of `256` in [explain.py](file:///c:/Users/zakar/Desktop/DATA/Credit-Card-Fraud-Detection-Using-Autoencoders/src/evaluation/explain.py) to keep memory near-zero.
4. **CI/CD & Docker Pipeline Repairs**: Configured the build workflow [.github/workflows/ci.yml](file:///c:/Users/zakar/Desktop/DATA/Credit-Card-Fraud-Detection-Using-Autoencoders/.github/workflows/ci.yml) and [Dockerfile](file:///c:/Users/zakar/Desktop/DATA/Credit-Card-Fraud-Detection-Using-Autoencoders/Dockerfile) to properly package FastAPI dependencies and use Python `urllib` for lightweight Docker health checks.
5. **Dashboard Features (Streamlit)**: Added tabs for **Adversarial Robustness Lab**, **Active Learning Loop**, and integrated direct SHAP horizontal bar charting into the detector.

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
