import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
import os
import matplotlib.pyplot as plt
from src.core.config import Config

class FraudEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_scores, threshold=None):
        """
        Calculate AUPRC, AUROC, and technical metrics.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)
        auroc = roc_auc_score(y_true, y_scores)
        
        if threshold is None:
            # Optimal threshold based on F1 in PR curve or similar
            # For simplicity, we use the mean+3std logic elsewhere, but here we can optimize
            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            threshold = thresholds[np.argmax(f1_scores)]

        y_pred = [1 if e > threshold else 0 for e in y_scores]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Financial Cost Model
        total_cost = (fp * Config.COST_FP) + (fn * Config.COST_FN)
        
        metrics = {
            'AUPRC': auprc,
            'AUROC': auroc,
            'Total Cost ($)': total_cost,
            'Recall': tp / (tp + fn),
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'FPR': fp / (fp + tn)
        }
        return metrics, threshold

    @staticmethod
    def calculate_business_impact(y_true, y_scores, amts, threshold):
        """
        Translates AUC into Dollars.
        Calculates how much money was saved vs lost.
        """
        y_pred = [1 if e > threshold else 0 for e in y_scores]
        
        # Money originally at risk (all frauds)
        total_fraud_amount = np.sum(amts[y_true == 1])
        
        # Money recovered (Correctly identified frauds)
        saved_amount = np.sum(amts[(y_true == 1) & (np.array(y_pred) == 1)])
        
        # Missed money (False Negatives)
        missed_amount = total_fraud_amount - saved_amount
        
        # Cost of False Positives (customer friction)
        friction_cost = np.sum(np.array(y_pred) == 1) * Config.COST_FP # Simple model
        
        impact = {
            'Total Fraud Value ($)': total_fraud_amount,
            'Value Saved ($)': saved_amount,
            'Value Lost ($)': missed_amount,
            'Detection Efficiency (%)': (saved_amount / total_fraud_amount * 100) if total_fraud_amount > 0 else 0,
            'Net Business Benefit ($)': saved_amount - friction_cost
        }
        return impact

    @staticmethod
    def plot_pr_curve(y_true, y_scores, model_name="Autoencoder"):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUPRC={auc(recall, precision):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        img_path = os.path.join(Config.VIZ_DIR, f"pr_curve_{model_name.lower()}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"PR curve saved to: {img_path}")
        
        return img_path
