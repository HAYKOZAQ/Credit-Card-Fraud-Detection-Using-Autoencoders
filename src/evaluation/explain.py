import torch
import numpy as np
import shap
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.core.config import Config

class FraudExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.model.eval()
        self.background_data = torch.FloatTensor(background_data).to(Config.DEVICE)
        self.explainer = None

    @staticmethod
    def reconstruction_error_wrapper(x_np, model):
        chunk_size = 256
        mses = []
        for i in range(0, len(x_np), chunk_size):
            chunk = x_np[i:i+chunk_size]
            x_tensor = torch.FloatTensor(chunk).to(Config.DEVICE)
            with torch.no_grad():
                if model.__class__.__name__ == 'GraphAutoencoder':
                    n_nodes = x_tensor.shape[0]
                    edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).to(Config.DEVICE)
                    outputs = model(x_tensor, edge_index)
                else:
                    outputs = model(x_tensor)
                    
                if isinstance(outputs, tuple):
                    recon_x = outputs[0]
                else:
                    recon_x = outputs
                
                mse = torch.mean((x_tensor - recon_x)**2, dim=1)
                mses.append(mse.cpu().numpy())
        return np.concatenate(mses)

    def _get_explainer(self):
        if self.explainer is None:
            background_sample = self.background_data[:50].cpu().numpy()
            self.explainer = shap.KernelExplainer(
                lambda x: self.reconstruction_error_wrapper(x, self.model), 
                background_sample
            )
        return self.explainer

    def explain_instance(self, instance, feature_names):
        """
        Explain a single fraudulent transaction.
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(instance.reshape(1, -1))[0]
        
        import time
        import seaborn as sns
        plot_path = os.path.join(Config.VIZ_DIR, f"shap_explanation_{int(time.time()*1000)}.png")
        
        # 1. Plot premium horizontal bar chart
        max_display = 10
        abs_vals = np.abs(shap_values)
        indices = np.argsort(abs_vals)[::-1]
        top_indices = indices[:max_display]
        
        top_features = [feature_names[i] for i in top_indices]
        top_shap = [shap_values[i] for i in top_indices]
        top_data = [instance[i] for i in top_indices]
        
        labels = []
        for f, val in zip(top_features, top_data):
            labels.append(f"{f} = {val:.2f}")
            
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        # Color coding: red increases anomaly score (positive SHAP), blue reduces it (negative SHAP)
        colors = ['#EF4444' if v > 0 else '#3B82F6' for v in top_shap]
        
        ax.barh(labels[::-1], top_shap[::-1], color=colors[::-1], edgecolor='none', height=0.6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.axvline(0, color='#9CA3AF', linestyle='-', linewidth=1.2)
        
        ax.set_title("🔬 Decision Explainer: Top Feature Contributions", fontsize=14, fontweight='bold', pad=20, color='#1F2937')
        ax.set_xlabel("SHAP Value (Contribution to Anomaly Score)", fontsize=11, labelpad=12, color='#4B5563')
        
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', transparent=True)
        plt.close()
        
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values,
            'Actual Value': instance
        }).sort_values('SHAP Value', ascending=False)
        
        csv_path = os.path.join(Config.RESULTS_DIR, "shap_explanation.csv")
        explanation_df.to_csv(csv_path, index=False)
        
        return plot_path, csv_path
