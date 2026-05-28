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
        self.baseline_stats_path = os.path.join(Config.MODEL_DIR, "stats_month0.pkl")
        self.baseline_stats = None
        
    def simulate_months(self, num_months=6):
        """
        Split data into approximately monthly segments.
        """
        # Use pandas split to ensure DataFrames are returned
        chunk_size = len(self.df) // num_months
        months = []
        for i in range(num_months):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_months - 1 else len(self.df)
            months.append(self.df.iloc[start_idx:end_idx].copy())
        return months

    def _save_baseline_artifacts(self):
        """Save month-0 scaler, encoders, and statistics so static eval uses correct references."""
        shutil.copy(Config.SCALER_PATH, self.baseline_scaler_path)
        shutil.copy(Config.ENCODER_PATH, self.baseline_encoder_path)
        if self.baseline_stats is not None:
            with open(self.baseline_stats_path, 'wb') as f:
                pickle.dump(self.baseline_stats, f)

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
        scaler_backup = Config.SCALER_PATH + ".drift_bak"
        encoder_backup = Config.ENCODER_PATH + ".drift_bak"
        had_scaler = os.path.exists(Config.SCALER_PATH)
        had_encoder = os.path.exists(Config.ENCODER_PATH)
        if had_scaler:
            shutil.copy(Config.SCALER_PATH, scaler_backup)
        if had_encoder:
            shutil.copy(Config.ENCODER_PATH, encoder_backup)

        preprocessor = Preprocessor()
        X_train, X_test, X_fraud, _, _, _ = preprocessor.fit_transform(df_month)
        
        # Store statistics from month 0 for use in static evaluation
        if month_idx == 0:
            self.baseline_stats = {
                'merchant_stats': preprocessor.merchant_stats,
                'category_stats': preprocessor.category_stats,
                'state_stats': preprocessor.state_stats
            }

        if month_idx == 0:
            self._save_baseline_artifacts()
        else:
            if had_scaler:
                shutil.copy(scaler_backup, Config.SCALER_PATH)
                os.remove(scaler_backup)
            if had_encoder:
                shutil.copy(encoder_backup, Config.ENCODER_PATH)
                os.remove(encoder_backup)
        
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
        # Use month-0 baseline scaler, encoders, and statistics
        with open(self.baseline_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(self.baseline_stats_path, 'rb') as f:
            baseline_stats = pickle.load(f)

        # Apply feature engineering using baseline statistics
        from src.core.data_loader import _engineer_features
        df_month = _engineer_features(
            df_month,
            merchant_stats=baseline_stats['merchant_stats'],
            category_stats=baseline_stats['category_stats'],
            state_stats=baseline_stats['state_stats']
        )
        
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
