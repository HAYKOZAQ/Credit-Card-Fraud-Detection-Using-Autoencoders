import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pickle
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
        
    def simulate_months(self, num_months=6):
        """
        Split data into approximately monthly segments.
        """
        months = np.array_split(self.df, num_months)
        return months

    def run_drift_experiment(self, months):
        print("\n>>> Starting Concept Drift Simulation <<<")
        results = []
        
        # Initial Training on Month 0
        m0 = months[0]
        print("Training Initial Model on Month 0...")
        initial_metrics = self._process_month(0, m0, train=True)
        results.append(initial_metrics)
        
        initial_model = get_model('standard')
        model_name = initial_model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
        initial_model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE))
        
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
        # fit_transform now returns 6 values
        X_train, X_test, X_fraud, _, _, _ = preprocessor.fit_transform(df_month)
        
        train_loader = DataLoader(FraudDataset(X_train), batch_size=Config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(FraudDataset(X_test), batch_size=Config.BATCH_SIZE)
        fraud_loader = DataLoader(FraudDataset(X_fraud), batch_size=Config.BATCH_SIZE)
        
        model = get_model('standard')
        if train:
            train_model(model, train_loader, test_loader)
        
        # Load best and evaluate
        model_name = model.__class__.__name__.lower()
        model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE))
        test_errors = get_reconstruction_errors(model, test_loader)
        fraud_errors = get_reconstruction_errors(model, fraud_loader)
        
        y_true = np.concatenate([np.zeros(len(test_errors)), np.ones(len(fraud_errors))])
        y_scores = np.concatenate([test_errors, fraud_errors])
        
        metrics, _ = FraudEvaluator.calculate_metrics(y_true, y_scores)
        metrics['Month'] = month_idx
        metrics['Method'] = 'Initial' if month_idx == 0 else 'Adaptive'
        return metrics

    def _evaluate_static_model(self, model, df_month):
        # Use initial scaler
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        # Preprocess features
        df_month['trans_date_trans_time'] = pd.to_datetime(df_month['trans_date_trans_time'])
        df_month['dob'] = pd.to_datetime(df_month['dob'])
        df_month['age'] = (df_month['trans_date_trans_time'] - df_month['dob']).dt.days // 365
        df_month['hour'] = df_month['trans_date_trans_time'].dt.hour
        df_month['distance_km'] = haversine_distance(df_month['lat'], df_month['long'],
                                                   df_month['merch_lat'], df_month['merch_long'])
        df_month['amt_log'] = np.log1p(df_month['amt'])
        
        # Drop columns
        features = Config.FEATURES
        
        # Load encoders (FIX: Use saved encoders instead of fitting new ones)
        with open(Config.ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
            
        for col in Config.CATEGORICAL_COLS:
            if col in encoders:
                df_month[col] = encoders[col].transform(df_month[col].astype(str))
            else:
                # Handle unseen labels by filling with a default or most frequent (simplified)
                df_month[col] = df_month[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
            
        X = df_month[features].values
        X_scaled = scaler.transform(X)
        y_true = df_month['is_fraud'].values
        
        loader = DataLoader(FraudDataset(X_scaled), batch_size=Config.BATCH_SIZE)
        errors = get_reconstruction_errors(model, loader)
        
        metrics, _ = FraudEvaluator.calculate_metrics(y_true, errors)
        return metrics
