from xgboost import XGBClassifier
from src.core.config import Config

class Baselines:
    @staticmethod
    def train_xgboost(X_train, y_train, X_test):
        print("Training XGBoost baseline...")
        ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        model = XGBClassifier(scale_pos_weight=ratio, random_state=Config.SEED, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        return model, preds, probs
