import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pickle
import shutil
from src.core.config import Config
from src.core.data_loader import Preprocessor, FraudDataset, haversine_distance
from src.core.model import get_model
from src.training.train import train_model
from src.evaluation.evaluate import get_reconstruction_errors
from src.core.metrics import FraudEvaluator
from torch.utils.data import DataLoader

class DriftSimulator:
    def __init__(self, data_path=Config.DATA_PATH):
        self.df = pd.read_csv(data_path)
        self.df['trans_date_trans_time'] = pd.to_datetime(self.df['trans_date_trans_time'])
        self.df = self.df.sort_values('trans_date_trans_time')
        self.baseline_scaler_path = os.path.join(Config.MODEL_DIR, "scaler_month0.pkl")
        self.baseline_encoder_path = os.path.join(Config.MODEL_DIR, "encoders_month0.pkl")
        
    def simulate_months(self, num_months=6):
        """
        Split data into approximately monthly segments.
        """
        months = np.array_split(self.df, num_months)
        return months

    def _save_baseline_artifacts(self):
        """Save month-0 scaler and encoders so static eval uses correct references."""
        shutil.copy(Config.SCALER_PATH, self.baseline_scaler_path)
        shutil.copy(Config.ENCODER_PATH, self.baseline_encoder_path)

    def run_drift_experiment(self, months):
        print("\n>>> Starting Concept Drift Simulation <<<")
        results = []
        
        # Initial Training on Month 0
        m0 = months[0]
        print("Training Initial Model on Month 0...")
        initial_metrics = self._process_month(0, m0, train=True)
        results.append(initial_metrics)
        self._save_baseline_artifacts()
        
        initial_model = get_model('standard')
        model_name = initial_model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}_month0.pth")
        initial_model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
        
        # Evaluate on subsequent months
        for i in range(1, len(months)):
            print(f"Propagating model to Month {i} (Zero-Shot) - Static...")
            month_data = months[i]
            metrics = self._evaluate_static_model(initial_model, month_data.copy())
            metrics['Month'] = i
            metrics['Method'] = 'Static Model'
            results.append(metrics)
            
            # Simulated Retraining (Adaptive)
            print(f"Retraining model on Month {i} (Adaptive)...")
            retrained_metrics = self._process_month(i, month_data, train=True)
            retrained_metrics['Method'] = 'Adaptive (Retrained)'
            results.append(retrained_metrics)

        # Finalize Results
        df_results = pd.DataFrame(results)
        print("\n" + "="*40)
        print("DRIFT SIMULATION REPORT")
        print("="*40)
        print(df_results[['Month', 'Method', 'AUPRC', 'AUROC']])
        
        # Matplotlib Version
        plt.figure(figsize=(10, 6))
        for method in df_results['Method'].unique():
            subset = df_results[df_results['Method'] == method]
            plt.plot(subset['Month'], subset['AUPRC'], marker='o', label=method)
        plt.title("Concept Drift Impact")
        plt.xlabel("Month")
        plt.ylabel("AUPRC")
        plt.legend()
        plt.grid(True)
        img_path = os.path.join(Config.VIZ_DIR, "drift_simulation.png")
        plt.savefig(img_path)
        plt.close()
        print(f"Drift plot saved to: {img_path}")
        
        csv_path = os.path.join(Config.RESULTS_DIR, "drift_metrics.csv")
        df_results.to_csv(csv_path, index=False)
        
        return img_path, csv_path

    def _process_month(self, month_idx, df_month, train=True):
        preprocessor = Preprocessor()
        X_train, X_test, X_fraud, _, _, _ = preprocessor.fit_transform(df_month)
        
        train_loader = DataLoader(FraudDataset(X_train), batch_size=Config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(FraudDataset(X_test), batch_size=Config.BATCH_SIZE)
        fraud_loader = DataLoader(FraudDataset(X_fraud), batch_size=Config.BATCH_SIZE)
        
        model = get_model('standard')
        if train:
            model_name = model.__class__.__name__.lower()
            save_name = f"{model_name}_month{month_idx}.pth"
            train_model(model, train_loader, test_loader, save_name=save_name)
        
        model_name = model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}_month{month_idx}.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
        test_errors = get_reconstruction_errors(model, test_loader)
        fraud_errors = get_reconstruction_errors(model, fraud_loader)
        
        y_true = np.concatenate([np.zeros(len(test_errors)), np.ones(len(fraud_errors))])
        y_scores = np.concatenate([test_errors, fraud_errors])
        
        metrics, _ = FraudEvaluator.calculate_metrics(y_true, y_scores)
        metrics['Month'] = month_idx
        metrics['Method'] = 'Initial' if month_idx == 0 else 'Adaptive'
        return metrics

    def _evaluate_static_model(self, model, df_month):
        # Use month-0 baseline scaler and encoders
        with open(self.baseline_scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        df_month['trans_date_trans_time'] = pd.to_datetime(df_month['trans_date_trans_time'])
        df_month['dob'] = pd.to_datetime(df_month['dob'])
        df_month['age'] = (df_month['trans_date_trans_time'] - df_month['dob']).dt.days // 365
        df_month['hour'] = df_month['trans_date_trans_time'].dt.hour
        df_month['distance_km'] = haversine_distance(df_month['lat'], df_month['long'],
                                                    df_month['merch_lat'], df_month['merch_long'])
        df_month['amt_log'] = np.log1p(df_month['amt'])
        
        features = Config.FEATURES
        
        with open(self.baseline_encoder_path, 'rb') as f:
            encoders = pickle.load(f)
            
        for col in Config.CATEGORICAL_COLS:
            le = encoders.get(col)
            if le is None:
                continue
            raw_vals = df_month[col].astype(str)
            known = set(le.classes_)
            safe_vals = raw_vals.apply(lambda x: x if x in known else le.classes_[0])
            df_month[col] = le.transform(safe_vals)
            
        X = df_month[features].values
        X_scaled = scaler.transform(X)
        y_true = df_month['is_fraud'].values
        
        loader = DataLoader(FraudDataset(X_scaled), batch_size=Config.BATCH_SIZE)
        errors = get_reconstruction_errors(model, loader)
        
        metrics, _ = FraudEvaluator.calculate_metrics(y_true, errors)
        return metrics
