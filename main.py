import torch
import random
import numpy as np
import pandas as pd
import plotly.express as px
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from src.core.config import Config
from src.core.data_loader import get_dataloaders, get_sequential_loaders
from src.core.model import get_model
from src.training.train import train_model
from src.evaluation.evaluate import get_reconstruction_errors, plot_history, get_uncertainty_scores, plot_uncertainty_vs_error, plot_latent_tsne, plot_merchant_heatmap, plot_cumulative_fraud, plot_feature_importance, calculate_shap_values, ensemble_scoring
from src.evaluation.baselines import Baselines
from src.core.metrics import FraudEvaluator
from src.evaluation.hybrid import HybridModel
from src.evaluation.simulation import DriftSimulator
from src.evaluation.robustness import evaluate_robustness
from src.training.active_learning import ActiveLearningLab
from src.core.data_loader import get_merchant_loaders
from src.utils.task_logger import TaskLogger

def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_background_sample(train_loader):
    """Save a small slice of training data for SHAP background."""
    data = train_loader.dataset.data.numpy()
    background_slice = data[:100]
    path = os.path.join(Config.MODEL_DIR, "background_sample.pkl")
    with open(path, 'wb') as f:
        pickle.dump(background_slice, f)
    print(f"SHAP background sample saved to {path}")

def run_research_suite(train_loader, test_loader, fraud_loader):
    results = []
    
    # Save background for SHAP
    save_background_sample(train_loader)
    
    # 1. XGBoost Baseline
    X_train_raw = train_loader.dataset.data.numpy()
    X_test_raw = test_loader.dataset.data.numpy()
    X_fraud_raw_arr = fraud_loader.dataset.data.numpy()
    
    n_train = min(len(X_train_raw), 5000)
    n_fraud = min(len(X_fraud_raw_arr), 1000)
    
    X_bench_train = np.concatenate([X_train_raw[:n_train], X_fraud_raw_arr[:n_fraud]]) 
    y_bench_train = np.concatenate([np.zeros(n_train), np.ones(n_fraud)])
    
    X_bench_test = np.concatenate([X_test_raw, X_fraud_raw_arr])
    y_bench_test = np.concatenate([np.zeros(len(X_test_raw)), np.ones(len(X_fraud_raw_arr))])

    _, _, xgb_probs = Baselines.train_xgboost(X_bench_train, y_bench_train, X_bench_test)
    xgb_metrics, _ = FraudEvaluator.calculate_metrics(y_bench_test, xgb_probs)
    xgb_metrics['Model'] = 'XGBoost Baseline'
    results.append(xgb_metrics)

    # 2. AE Experiments
    for ae_type in ['standard', 'vae', 'denoising', 'contrastive', 'graph']:
        print(f"\n>>> Running Experiment: {ae_type.upper()} <<<")
        model = get_model(ae_type)
        history = train_model(model, train_loader, test_loader)
        
        model_name = model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_save_path))
        
        # Save training history plots
        plot_history(history)
        
        test_errors = get_reconstruction_errors(model, test_loader)
        fraud_errors = get_reconstruction_errors(model, fraud_loader)
        y_true = [0] * len(test_errors) + [1] * len(fraud_errors)
        y_scores = np.concatenate([test_errors, fraud_errors])
        
        m_results, opt_thresh = FraudEvaluator.calculate_metrics(y_true, y_scores)
        m_results['Model'] = f'AE ({ae_type})'
        results.append(m_results)
        
        # Generate Precision-Recall Curves
        FraudEvaluator.plot_pr_curve(y_true, y_scores, model_name=ae_type)
        
        # Generate Feature Importance for the best model (Standard as baseline)
        if ae_type == 'standard':
            plot_feature_importance(model, fraud_loader, Config.FEATURES)

    # 3. Final Comparison
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RESEARCH REPORT SUMMARY")
    print("="*60)
    print(df_results[['Model', 'AUPRC', 'AUROC', 'Total Cost ($)']])
    
    csv_path = os.path.join(Config.RESULTS_DIR, "research_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    
    csv_path = os.path.join(Config.RESULTS_DIR, "research_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    
    # Matplotlib Comparison
    plt.figure(figsize=(10, 6))
    df_results_sorted = df_results.sort_values('AUPRC', ascending=False)
    plt.bar(df_results_sorted['Model'], df_results_sorted['AUPRC'], color='skyblue')
    plt.title("Research Suite Comparison")
    plt.ylabel("AUPRC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_path = os.path.join(Config.VIZ_DIR, "research_comparison.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Comparison plot saved to: {img_path}")
    
    return df_results

def run_advanced_suite(train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df):
    results = []
    
    # 1. Uncertainty Quantification
    print("\n>>> Task 1: Uncertainty Quantification (MC Dropout) <<<")
    mc_model = get_model('mc_dropout')
    print("Training MC Dropout Model...")
    train_model(mc_model, train_loader, test_loader)
    
    # Load best weights
    mc_path = os.path.join(Config.MODEL_DIR, "mcdropoutautoencoder.pth")
    mc_model.load_state_dict(torch.load(mc_path))
    
    # Evaluate Uncertainty
    print("Evaluating Uncertainty...")
    test_errors, test_vars = get_uncertainty_scores(mc_model, test_loader)
    fraud_errors, fraud_vars = get_uncertainty_scores(mc_model, fraud_loader)
    
    plot_uncertainty_vs_error(
        np.concatenate([test_errors, fraud_errors]),
        np.concatenate([test_vars, fraud_vars]),
        np.concatenate([np.zeros(len(test_errors)), np.ones(len(fraud_errors))])
    )
    
    # 2. Adversarial Robustness
    print("\n>>> Task 2: Adversarial Robustness <<<")
    
    # Train Denoising
    print("Training Denoising AE...")
    denoising_model = get_model('denoising')
    train_model(denoising_model, train_loader, test_loader)
    
    # Load best weights
    denoise_path = os.path.join(Config.MODEL_DIR, "denoisingautoencoder.pth")
    denoising_model.load_state_dict(torch.load(denoise_path))
    
    # Robustness Eval
    rob_results = evaluate_robustness({
        'MC Dropout': mc_model, 
        'Denoising': denoising_model
    }, test_loader, epsilon=0.1)
    
    rob_df = pd.DataFrame(list(rob_results.items()), columns=['Model', 'Reconstruction Error (MSE)'])
    rob_path = os.path.join(Config.RESULTS_DIR, "robustness_results.csv")
    rob_df.to_csv(rob_path, index=False)
    print(f"Robustness results saved to {rob_path}")
    
    # 3. Multi-scale Anomaly Detection
    print("\n>>> Task 3: Multi-scale Detection <<<")
    # Transaction level: MC Model (already trained)
    
    # Card level
    print("Training LSTM (Card-level)...")
    train_seq, test_seq, fraud_seq = get_sequential_loaders()
    lstm_model = get_model('lstm')
    train_model(lstm_model, train_seq, test_seq)
    
    # Load best weights
    lstm_path = os.path.join(Config.MODEL_DIR, "lstmautoencoder.pth")
    lstm_model.load_state_dict(torch.load(lstm_path))

    # Merchant level
    print("Training Merchant AE...")
    merch_train, merch_test = get_merchant_loaders()
    merch_model = get_model('lstm') # Use LSTM for merchant sequences too
    train_model(merch_model, merch_train, merch_test)
    
    # 4. Active Learning
    print("\n>>> Task 4: Active Learning Loop <<<")
    X_test_scaled = test_loader.dataset.data.numpy()
    X_fraud_scaled = fraud_loader.dataset.data.numpy()
    
    pool_X = np.concatenate([X_test_scaled, X_fraud_scaled])
    pool_y = np.concatenate([np.zeros(len(X_test_scaled)), np.ones(len(X_fraud_scaled))])
    
    # Shuffle pool
    idx = np.arange(len(pool_X))
    np.random.shuffle(idx)
    pool_X = pool_X[idx]
    pool_y = pool_y[idx]
    
    # Split
    split = int(len(pool_X) * 0.5)
    al_X = pool_X[:split]
    al_y = pool_y[:split]
    eval_X = pool_X[split:]
    eval_y = pool_y[split:]
    
    lab = ActiveLearningLab(mc_model, al_X, al_y, eval_X, eval_y)
    al_results = lab.run_simulation(strategy='reconstruction_error', budget=200, step_size=20)
    
    al_path = os.path.join(Config.RESULTS_DIR, "active_learning_results.csv")
    al_results.to_csv(al_path, index=False)
    print(f"Active Learning results saved to {al_path}")
    
    # 5. Visualizations
    print("\n>>> Task 5: Visualizations <<<")
    plot_latent_tsne(mc_model, test_loader, fraud_loader)
    
    # Merchant Heatmap
    f_errors = get_reconstruction_errors(mc_model, fraud_loader)
    plot_merchant_heatmap(fraud_df, f_errors)
    
    # Cumulative Fraud
    plot_cumulative_fraud(mc_model, test_df, fraud_df, test_loader, fraud_loader)
    
    print("Advanced suite complete.")

def run_interpretability_suite(train_loader, test_loader, fraud_loader):
    print("\n>>> Interpretability Suite: SHAP & Feature Attribution <<<")
    model = get_model('standard')
    model_name = model.__class__.__name__.lower()
    model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
    
    # Load best trained model if exists, else train
    if os.path.exists(model_save_path):
        print(f"Loading existing weights for {model_name}...")
        model.load_state_dict(torch.load(model_save_path))
    else:
        train_model(model, train_loader, test_loader)
        model.load_state_dict(torch.load(model_save_path))
    
    features = Config.FEATURES
    calculate_shap_values(model, train_loader, test_loader, features)

def run_ensemble_suite(train_loader, test_loader, fraud_loader):
    print("\n>>> Ensemble Suite: Meta-Model Scoring <<<")
    # Load or train few basic experts
    experts = {}
    for m_type in ['standard', 'vae', 'attention_ae']:
        model = get_model(m_type)
        train_model(model, train_loader, test_loader)
        experts[m_type] = model
    
    test_scores = ensemble_scoring(experts, test_loader)
    fraud_scores = ensemble_scoring(experts, fraud_loader)
    
    y_true = [0] * len(test_scores) + [1] * len(fraud_scores)
    y_scores = np.concatenate([test_scores, fraud_scores])
    
    metrics, _ = FraudEvaluator.calculate_metrics(y_true, y_scores)
    metrics['Model'] = 'Ensemble'
    
    ens_df = pd.DataFrame([metrics])
    ens_path = os.path.join(Config.RESULTS_DIR, "ensemble_metrics.csv")
    ens_df.to_csv(ens_path, index=False)
    
    print(f"Ensemble metrics saved to {ens_path}")
    FraudEvaluator.plot_pr_curve(y_true, y_scores, model_name="Ensemble_Experts")

def main():
    parser = argparse.ArgumentParser(description="FraudGuard Research & Simulation Suite")
    parser.add_argument('--mode', type=str, default='advanced', 
                        choices=['research', 'simulation', 'advanced', 'interpret', 'ensemble', 'ablation'], 
                        help='Choose execution mode')
    args = parser.parse_args()
    
    logger = TaskLogger()
    logger.start_task(f"Execution Mode: {args.mode}")

    set_seed()
    
    if args.mode == 'research':
        train_loader, test_loader, fraud_loader, _, _, _, _ = get_dataloaders()
        run_research_suite(train_loader, test_loader, fraud_loader)
        print(f"\nResults saved in {Config.RESULTS_DIR} and {Config.VIZ_DIR}")
        logger.end_task(f"Execution Mode: {args.mode}", success=True)
        
    elif args.mode == 'advanced':
        train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df = get_dataloaders()
        run_advanced_suite(train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df)
        logger.end_task(f"Execution Mode: {args.mode}", success=True)
        
    elif args.mode == 'simulation':
        sim = DriftSimulator()
        months = sim.simulate_months(num_months=4)
        viz_path, csv_path = sim.run_drift_experiment(months)
        print(f"\nSimulation complete!")
        print(f"\nSimulation complete!")
        print(f"Drift Visual: {viz_path}")
        print(f"Drift Log: {csv_path}")

    elif args.mode == 'interpret':
        train_loader, test_loader, fraud_loader, _, _, _, _ = get_dataloaders()
        run_interpretability_suite(train_loader, test_loader, fraud_loader)

    elif args.mode == 'ensemble':
        train_loader, test_loader, fraud_loader, _, _, _, _ = get_dataloaders()
        run_ensemble_suite(train_loader, test_loader, fraud_loader)

    elif args.mode == 'ablation':
        from ablation_study import run_ablation
        run_ablation()

if __name__ == "__main__":
    main()
