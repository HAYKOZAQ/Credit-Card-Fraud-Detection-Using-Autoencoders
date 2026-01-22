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
        
        # We need a wrapper function for SHAP that returns the reconstruction error
        # since standard SHAP expects single-output regression or classification.
        # Here we explain 'Total Reconstruction Error'.
    
    @staticmethod
    def reconstruction_error_wrapper(x_np, model):
        x_tensor = torch.FloatTensor(x_np).to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, tuple): # VAE
                recon_x = outputs[0]
            else:
                recon_x = outputs
            
            # MSE per row
            mse = torch.mean((x_tensor - recon_x)**2, dim=1)
        return mse.cpu().numpy()

    def explain_instance(self, instance, feature_names):
        """
        Explain a single fraudulent transaction.
        """
        # SHAP KernelExplainer is slow but model-agnostic
        # We'll use a small background sample for speed
        background_sample = self.background_data[:50].cpu().numpy()
        
        explainer = shap.KernelExplainer(
            lambda x: self.reconstruction_error_wrapper(x, self.model), 
            background_sample
        )
        
        shap_values = explainer.shap_values(instance.reshape(1, -1))
        
        # Plotting
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            instance, 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plot_path = os.path.join(Config.VIZ_DIR, "shap_explanation.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save summary to results
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0],
            'Actual Value': instance
        }).sort_values('SHAP Value', ascending=False)
        
        csv_path = os.path.join(Config.RESULTS_DIR, "shap_explanation.csv")
        explanation_df.to_csv(csv_path, index=False)
        
        return plot_path, csv_path
