from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from src.core.config import Config
import pandas as pd

class Baselines:
    @staticmethod
    def train_logistic_regression(X_train, y_train, X_test):
        print("Training Logistic Regression baseline...")
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=Config.SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        return model, preds, probs

    @staticmethod
    def train_xgboost(X_train, y_train, X_test):
        print("Training XGBoost baseline...")
        # Handle imbalance with scale_pos_weight
        ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        model = XGBClassifier(scale_pos_weight=ratio, random_state=Config.SEED, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        return model, preds, probs
