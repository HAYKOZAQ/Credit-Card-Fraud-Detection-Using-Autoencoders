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
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(Config.DEVICE)
                if hasattr(self.ae, 'encoder'):
                    z = self.ae.encoder(inputs)
                elif hasattr(self.ae, 'encode'):
                    z, _ = self.ae.encode(inputs)
                elif hasattr(self.ae, 'gc1'):
                    dists = torch.cdist(inputs, inputs)
                    sigma = dists.mean()
                    adj = torch.exp(-dists / (sigma + 1e-8))
                    adj = adj / adj.sum(dim=1, keepdim=True)
                    h = torch.relu(self.ae.gc1(inputs, adj))
                    z = torch.relu(self.ae.gc2(h, adj))
                else:
                    raise TypeError(f"Unsupported model type: {type(self.ae).__name__}")
                embeddings.append(z.cpu().numpy())
        return np.concatenate(embeddings)

    def train_classifier(self, X_train_emb, y_train, X_test_emb):
        print("Training Hybrid Classifier (AE Embeddings + XGBoost)...")
        ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        self.classifier = XGBClassifier(scale_pos_weight=ratio, random_state=Config.SEED, use_label_encoder=False, eval_metric='logloss')
        self.classifier.fit(X_train_emb, y_train)
        preds = self.classifier.predict(X_test_emb)
        probs = self.classifier.predict_proba(X_test_emb)[:, 1]
        return preds, probs
