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
        x_tensor = torch.FloatTensor(x_np).to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, tuple):
                recon_x = outputs[0]
            else:
                recon_x = outputs
            
            mse = torch.mean((x_tensor - recon_x)**2, dim=1)
        return mse.cpu().numpy()

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
        shap_values = explainer.shap_values(instance.reshape(1, -1))
        
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
        try:
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                instance, 
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(plot_path)
        except TypeError:
            shap.plots.force(explainer.expected_value, shap_values[0], instance,
                           feature_names=feature_names, show=False)
            plt.savefig(plot_path)
        plt.close()
        
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0],
            'Actual Value': instance
        }).sort_values('SHAP Value', ascending=False)
        
        csv_path = os.path.join(Config.RESULTS_DIR, "shap_explanation.csv")
        explanation_df.to_csv(csv_path, index=False)
        
        return plot_path, csv_path
