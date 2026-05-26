import pandas as pd
import numpy as np
import torch
import os
from src.core.config import Config
from src.core.data_loader import get_dataloaders
from src.core.model import get_model
from src.training.train import train_model
from src.evaluation.evaluate import get_reconstruction_errors
from src.core.metrics import FraudEvaluator

def run_ablation():
    print(">>> Starting Ablation Study <<<")
    
    # 1. Baseline: Standard AE
    # 2. VAE (Probabilistic)
    # 3. Attention (Context-aware)
    # 4. Contrastive (Robust Latents)
    
    models_to_test = ['standard', 'vae', 'attention_ae', 'contrastive']
    results = []
    
    train_loader, test_loader, fraud_loader, _, _, _, _ = get_dataloaders()
    
    for m_type in models_to_test:
        print(f"\nTesting Component: {m_type}")
        model = get_model(m_type)
        history = train_model(model, train_loader, test_loader)
        
        # Load best weights (saved in train_model)
        model_name = model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
        
        test_errors = get_reconstruction_errors(model, test_loader)
        fraud_errors = get_reconstruction_errors(model, fraud_loader)
        
        y_true = [0] * len(test_errors) + [1] * len(fraud_errors)
        y_scores = np.concatenate([test_errors, fraud_errors])
        
        metrics, _ = FraudEvaluator.calculate_metrics(y_true, y_scores)
        metrics['Component'] = m_type
        results.append(metrics)
        
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("ABLATION STUDY SUMMARY")
    print("="*40)
    print(df[['Component', 'AUPRC', 'AUROC', 'Total Cost ($)']])
    
    csv_path = os.path.join(Config.RESULTS_DIR, "ablation_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAblation study results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
