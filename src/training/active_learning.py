import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from src.evaluation.evaluate import get_reconstruction_errors
from src.core.config import Config

class ActiveLearningLab:
    def __init__(self, autoencoder, X_unlabeled, y_unlabeled, X_test, y_test):
        self.ae = autoencoder
        self.X_pool = X_unlabeled
        self.y_pool = y_unlabeled
        self.X_test = X_test
        self.y_test = y_test
        
        self.results = {'step': [], 'auprc': [], 'strategy': []}
        
    def get_ae_errors(self):
        # Create loader
        dataset = TensorDataset(torch.FloatTensor(self.X_pool), torch.FloatTensor(self.X_pool))
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        return get_reconstruction_errors(self.ae, loader)
        
    def run_simulation(self, strategy='reconstruction_error', budget=100, step_size=10):
        labeled_mask = np.zeros(len(self.X_pool), dtype=bool)
        current_indices = []
        
        print(f"Starting Active Learning Simulation ({strategy})...")
        
        # Pre-calculate errors if needed
        errors = None
        if strategy == 'reconstruction_error':
            errors = self.get_ae_errors()
        
        for step in range(0, budget, step_size):
            # Select samples
            if strategy == 'reconstruction_error':
                # Filter out already labeled
                candidates_idx = np.where(~labeled_mask)[0]
                candidates_errors = errors[candidates_idx]
                
                # Top k errors
                # argsort gives ascending, so take last k
                if len(candidates_errors) < step_size:
                    new_indices = candidates_idx
                else:
                    top_k_local = np.argsort(candidates_errors)[-step_size:]
                    new_indices = candidates_idx[top_k_local]
                
            elif strategy == 'random':
                candidates_idx = np.where(~labeled_mask)[0]
                if len(candidates_idx) < step_size:
                    new_indices = candidates_idx
                else:
                    new_indices = np.random.choice(candidates_idx, step_size, replace=False)
            
            # Label them
            labeled_mask[new_indices] = True
            current_indices.extend(new_indices)
            
            # Train Supervised Model
            X_train = self.X_pool[current_indices]
            y_train = self.y_pool[current_indices]
            
            # Check if we have both classes
            if len(np.unique(y_train)) < 2:
                # Can't train reasonably, score 0
                score = 0
            else:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_train, y_train)
                # Predict
                probs = clf.predict_proba(self.X_test)[:, 1]
                score = average_precision_score(self.y_test, probs)
            
            self.results['step'].append(len(current_indices))
            self.results['auprc'].append(score)
            self.results['strategy'].append(strategy)
            
            print(f"Step {step+step_size}: Labeled {len(current_indices)}, AUPRC: {score:.4f}, Frauds Found: {y_train.sum()}")
            
        return pd.DataFrame(self.results)
