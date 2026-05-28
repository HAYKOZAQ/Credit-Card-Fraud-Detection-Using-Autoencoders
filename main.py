import torch
import random
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import matplotlib.pyplot as plt

from src.core.config import Config
from src.core.data_loader import get_dataloaders, get_sequential_loaders, get_merchant_loaders
from src.core.model import get_model
from src.training.train import train_model
from src.evaluation.evaluate import get_reconstruction_errors, plot_history, get_uncertainty_scores, plot_uncertainty_vs_error, plot_latent_tsne, plot_merchant_heatmap, plot_cumulative_fraud, plot_feature_importance, calculate_shap_values, ensemble_scoring
from src.evaluation.baselines import Baselines
from src.evaluation.hybrid import HybridModel
from src.core.metrics import FraudEvaluator
from src.evaluation.simulation import DriftSimulator
from src.evaluation.robustness import evaluate_robustness
from src.training.active_learning import ActiveLearningLab
from src.utils.task_logger import TaskLogger

_dataloader_cache = {"data": None, "failed": False}

def _get_cached_dataloaders():
    if _dataloader_cache["data"] is None:
        if _dataloader_cache["failed"]:
            _dataloader_cache["failed"] = False
        try:
            _dataloader_cache["data"] = get_dataloaders()
        except Exception as e:
            _dataloader_cache["failed"] = True
            raise RuntimeError(f"Failed to load data: {e}. Retry by calling again.") from e
    return _dataloader_cache["data"]

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
    
    n_normal_train = min(len(X_train_raw), 5000)
    n_fraud_train = min(len(X_fraud_raw_arr) // 2, 1000)
    n_fraud_train = max(n_fraud_train, 1) if len(X_fraud_raw_arr) > 0 else 0
    
    if n_fraud_train > 0:
        X_bench_train = np.concatenate([X_train_raw[:n_normal_train], X_fraud_raw_arr[:n_fraud_train]]) 
        y_bench_train = np.concatenate([np.zeros(n_normal_train), np.ones(n_fraud_train)])
        X_bench_test = np.concatenate([X_test_raw, X_fraud_raw_arr[n_fraud_train:]])
        y_bench_test = np.concatenate([np.zeros(len(X_test_raw)), np.ones(len(X_fraud_raw_arr) - n_fraud_train)])
    else:
        X_bench_train = X_train_raw[:n_normal_train]
        y_bench_train = np.zeros(n_normal_train)
        X_bench_test = X_test_raw
        y_bench_test = np.zeros(len(X_test_raw))
    
    n_pos_train = np.sum(y_bench_train)
    n_neg_train = len(y_bench_train) - n_pos_train
    if n_pos_train == 0 or n_neg_train == 0:
        print(f"Warning: XGBoost train set imbalance — pos={n_pos_train}, neg={n_neg_train}. Metrics may be degenerate.")

    _, _, xgb_probs = Baselines.train_xgboost(X_bench_train, y_bench_train, X_bench_test)
    xgb_metrics, _ = FraudEvaluator.calculate_metrics(y_bench_test, xgb_probs)
    xgb_metrics['Model'] = 'XGBoost Baseline'
    results.append(xgb_metrics)

    # 2. AE Experiments
    for ae_type in ['standard', 'vae', 'denoising', 'contrastive', 'graph']:
        print(f"\n>>> Running Experiment: {ae_type.upper()} <<<")
        model = get_model(ae_type)
        
        if ae_type == 'graph':
            try:
                from src.core.data_loader import get_pyg_loaders
                pyg_train, pyg_test, pyg_fraud = get_pyg_loaders()
                history = train_model(model, pyg_train, pyg_test)
                current_test_loader = pyg_test
                current_fraud_loader = pyg_fraud
            except ImportError:
                print("Skipping Graph Autoencoder (torch-geometric not installed).")
                continue
            except Exception as e:
                print(f"Skipping Graph Autoencoder ({e}).")
                continue
        else:
            history = train_model(model, train_loader, test_loader)
            current_test_loader = test_loader
            current_fraud_loader = fraud_loader
        
        model_name = model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
        
        # Save training history plots
        plot_history(history, model_name=ae_type)
        
        test_errors = get_reconstruction_errors(model, current_test_loader)
        fraud_errors = get_reconstruction_errors(model, current_fraud_loader)
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

    # 2.5 Hybrid Model Experiment
    print("\n>>> Running Experiment: HYBRID (Standard AE Embeddings + XGBoost) <<<")
    std_model = get_model('standard')
    std_save_path = os.path.join(Config.MODEL_DIR, "autoencoder.pth")
    if os.path.exists(std_save_path):
        std_model.load_state_dict(torch.load(std_save_path, map_location=Config.DEVICE, weights_only=False))
        hybrid = HybridModel(std_model)
        
        from torch.utils.data import DataLoader, TensorDataset
        bench_train_dataset = TensorDataset(torch.FloatTensor(X_bench_train), torch.FloatTensor(X_bench_train))
        bench_train_loader = DataLoader(bench_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        bench_test_dataset = TensorDataset(torch.FloatTensor(X_bench_test), torch.FloatTensor(X_bench_test))
        bench_test_loader = DataLoader(bench_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        X_train_emb = hybrid.get_embeddings(bench_train_loader)
        X_test_emb = hybrid.get_embeddings(bench_test_loader)
        
        _, hybrid_probs = hybrid.train_classifier(X_train_emb, y_bench_train, X_test_emb)
        hybrid_metrics, _ = FraudEvaluator.calculate_metrics(y_bench_test, hybrid_probs)
        hybrid_metrics['Model'] = 'Hybrid AE+XGB'
        results.append(hybrid_metrics)
        print(f"Hybrid Model Metrics: AUPRC={hybrid_metrics['AUPRC']:.4f}, AUROC={hybrid_metrics['AUROC']:.4f}, Cost=${hybrid_metrics['Total Cost ($)']}")

    # 3. Final Comparison
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RESEARCH REPORT SUMMARY")
    print("="*60)
    print(df_results[['Model', 'AUPRC', 'AUROC', 'Total Cost ($)']])
    
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
    # 1. Uncertainty Quantification
    print("\n>>> Task 1: Uncertainty Quantification (MC Dropout) <<<")
    mc_model = get_model('mc_dropout')
    print("Training MC Dropout Model...")
    train_model(mc_model, train_loader, test_loader)
    
    # Load best weights
    mc_path = os.path.join(Config.MODEL_DIR, "mcdropoutautoencoder.pth")
    if not os.path.exists(mc_path):
        raise FileNotFoundError(f"MC Dropout model not found at {mc_path}. Training may have failed.")
    mc_model.load_state_dict(torch.load(mc_path, map_location=Config.DEVICE, weights_only=False))
    
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
    denoising_model.load_state_dict(torch.load(denoise_path, map_location=Config.DEVICE, weights_only=False))
    
    # Robustness Eval
    rob_results = evaluate_robustness({
        'MC Dropout': mc_model, 
        'Denoising': denoising_model
    }, test_loader, epsilon=0.1)
    
    rob_rows = []
    for model_name, metrics in rob_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        rob_rows.append(row)
    rob_df = pd.DataFrame(rob_rows)
    rob_path = os.path.join(Config.RESULTS_DIR, "robustness_results.csv")
    rob_df.to_csv(rob_path, index=False)
    print(f"Robustness results saved to {rob_path}")
    
    # 3. Multi-scale Anomaly Detection
    print("\n>>> Task 3: Multi-scale Detection <<<")
    # Transaction level: MC Model (already trained)
    
    # Card level
    print("Training LSTM (Card-level)...")
    if not os.path.exists(Config.SCALER_PATH) or not os.path.exists(Config.ENCODER_PATH):
        print("Warning: Scaler/encoder files missing. Run --mode research first. Skipping LSTM training.")
    else:
        train_seq, test_seq, fraud_seq = get_sequential_loaders()
        lstm_model = get_model('lstm')
        train_model(lstm_model, train_seq, test_seq)
        
        lstm_path = os.path.join(Config.MODEL_DIR, "lstmautoencoder.pth")
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=Config.DEVICE, weights_only=False))
        
        # Evaluate LSTM
        print("Evaluating Card-level LSTM Autoencoder...")
        test_lstm_errors = get_reconstruction_errors(lstm_model, test_seq)
        fraud_lstm_errors = get_reconstruction_errors(lstm_model, fraud_seq)
        y_true_lstm = [0] * len(test_lstm_errors) + [1] * len(fraud_lstm_errors)
        y_scores_lstm = np.concatenate([test_lstm_errors, fraud_lstm_errors])
        lstm_metrics, _ = FraudEvaluator.calculate_metrics(y_true_lstm, y_scores_lstm)
        print(f"Card-level LSTM AE Metrics: AUPRC={lstm_metrics['AUPRC']:.4f}, AUROC={lstm_metrics['AUROC']:.4f}, Cost=${lstm_metrics['Total Cost ($)']}")
        
        # Save sequence metrics to csv
        seq_df = pd.DataFrame([{
            'Model': 'LSTM Card-level',
            'AUPRC': lstm_metrics['AUPRC'],
            'AUROC': lstm_metrics['AUROC'],
            'Total Cost ($)': lstm_metrics['Total Cost ($)'],
            'Recall': lstm_metrics['Recall'],
            'Precision': lstm_metrics['Precision']
        }])
        seq_csv_path = os.path.join(Config.RESULTS_DIR, "sequence_metrics.csv")
        seq_df.to_csv(seq_csv_path, index=False)
        print(f"Sequence metrics saved to {seq_csv_path}")

        # Train Transformer
        print("Training Transformer (Card-level)...")
        transformer_model = get_model('transformer')
        train_model(transformer_model, train_seq, test_seq, save_name="transformerautoencoder.pth")
        
        transformer_path = os.path.join(Config.MODEL_DIR, "transformerautoencoder.pth")
        transformer_model.load_state_dict(torch.load(transformer_path, map_location=Config.DEVICE, weights_only=False))
        
        # Evaluate Transformer
        print("Evaluating Card-level Transformer Autoencoder...")
        test_trans_errors = get_reconstruction_errors(transformer_model, test_seq)
        fraud_trans_errors = get_reconstruction_errors(transformer_model, fraud_seq)
        y_scores_trans = np.concatenate([test_trans_errors, fraud_trans_errors])
        trans_metrics, _ = FraudEvaluator.calculate_metrics(y_true_lstm, y_scores_trans)
        print(f"Card-level Transformer AE Metrics: AUPRC={trans_metrics['AUPRC']:.4f}, AUROC={trans_metrics['AUROC']:.4f}, Cost=${trans_metrics['Total Cost ($)']}")
        
        # Append Transformer metrics
        trans_df = pd.DataFrame([{
            'Model': 'Transformer Card-level',
            'AUPRC': trans_metrics['AUPRC'],
            'AUROC': trans_metrics['AUROC'],
            'Total Cost ($)': trans_metrics['Total Cost ($)'],
            'Recall': trans_metrics['Recall'],
            'Precision': trans_metrics['Precision']
        }])
        seq_df = pd.concat([seq_df, trans_df], ignore_ignore=True) if 'ignore_index' not in globals() else pd.concat([seq_df, trans_df], ignore_index=True)
        seq_df.to_csv(seq_csv_path, index=False)
        print(f"Transformer metrics appended to {seq_csv_path}")

    # Merchant level
    print("Training Merchant AE...")
    try:
        merch_train, merch_test = get_merchant_loaders()
        merch_model = get_model('lstm')
        train_model(merch_model, merch_train, merch_test, save_name="merchant_lstmautoencoder.pth")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not create merchant sequences ({e}). Skipping merchant training.")
    
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
    t_errors = get_reconstruction_errors(mc_model, test_loader)
    plot_cumulative_fraud(mc_model, test_df, fraud_df, test_loader, fraud_loader,
                         normal_errors=t_errors, fraud_errors=f_errors)
    
    print("Advanced suite complete.")

def run_interpretability_suite(train_loader, test_loader, fraud_loader):
    print("\n>>> Interpretability Suite: SHAP & Feature Attribution <<<")
    model = get_model('standard')
    model_name = model.__class__.__name__.lower()
    model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
    
    # Load best trained model if exists, else train
    if os.path.exists(model_save_path):
        print(f"Loading existing weights for {model_name}...")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
    else:
        train_model(model, train_loader, test_loader)
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
        else:
            print(f"Warning: model file {model_save_path} not saved after training.")
    
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
    start_time = logger.start_task(f"Execution Mode: {args.mode}")

    set_seed()
    
    if args.mode == 'research':
        train_loader, test_loader, fraud_loader, _, _, _, _ = _get_cached_dataloaders()
        run_research_suite(train_loader, test_loader, fraud_loader)
        print(f"\nResults saved in {Config.RESULTS_DIR} and {Config.VIZ_DIR}")
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)
        
    elif args.mode == 'advanced':
        train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df = _get_cached_dataloaders()
        run_advanced_suite(train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df)
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)
        
    elif args.mode == 'simulation':
        sim = DriftSimulator()
        months = sim.simulate_months(num_months=4)
        viz_path, csv_path = sim.run_drift_experiment(months)
        print(f"\nSimulation complete!")
        print(f"Drift Visual: {viz_path}")
        print(f"Drift Log: {csv_path}")
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)

    elif args.mode == 'interpret':
        train_loader, test_loader, fraud_loader, _, _, _, _ = _get_cached_dataloaders()
        run_interpretability_suite(train_loader, test_loader, fraud_loader)
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)

    elif args.mode == 'ensemble':
        train_loader, test_loader, fraud_loader, _, _, _, _ = _get_cached_dataloaders()
        run_ensemble_suite(train_loader, test_loader, fraud_loader)
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)

    elif args.mode == 'ablation':
        from ablation_study import run_ablation
        run_ablation()
        logger.end_task(f"Execution Mode: {args.mode}", start_time=start_time)

if __name__ == "__main__":
    main()
