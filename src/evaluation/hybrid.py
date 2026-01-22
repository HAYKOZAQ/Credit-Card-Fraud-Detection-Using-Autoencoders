import torch
import numpy as np
from xgboost import XGBClassifier
from src.core.config import Config

class HybridModel:
    def __init__(self, autoencoder):
        self.ae = autoencoder
        self.classifier = None

    def get_embeddings(self, data_loader):
        self.ae.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(Config.DEVICE)
                # Standard AE encoder
                if hasattr(self.ae, 'encoder'):
                    z = self.ae.encoder(inputs)
                else: # For VAE
                    z, _ = self.ae.encode(inputs)
                embeddings.append(z.cpu().numpy())
                # Note: For targets, we might need actual binary labels if available
                # However, during representation learning, we use whatever is in the loader
        return np.concatenate(embeddings)

    def train_classifier(self, X_train_emb, y_train, X_test_emb):
        print("Training Hybrid Classifier (AE Embeddings + XGBoost)...")
        ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        self.classifier = XGBClassifier(scale_pos_weight=ratio, random_state=Config.SEED)
        self.classifier.fit(X_train_emb, y_train)
        preds = self.classifier.predict(X_test_emb)
        probs = self.classifier.predict_proba(X_test_emb)[:, 1]
        return preds, probs
