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
        If threshold is not provided, uses mean+3*std of normal samples (no test leakage).
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        if len(y_true) == 0 or len(y_scores) == 0:
            return {
                'AUPRC': 0.0, 'AUROC': 0.0, 'Total Cost ($)': 0,
                'Recall': 0.0, 'Precision': 0.0, 'FPR': 0.0
            }, 0.0
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)
        auroc = roc_auc_score(y_true, y_scores)
        
        if threshold is None:
            normal_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
            if len(normal_scores) > 0:
                threshold = np.mean(normal_scores) + 3 * np.std(normal_scores)
            else:
                threshold = np.percentile(y_scores, 95)

        y_pred = [1 if e > threshold else 0 for e in y_scores]
        labels_present = sorted(set(list(y_true) + list(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
            tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
            fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
            fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
        
        # Financial Cost Model
        total_cost = (fp * Config.COST_FP) + (fn * Config.COST_FN)
        
        metrics = {
            'AUPRC': auprc,
            'AUROC': auroc,
            'Total Cost ($)': total_cost,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0
        }
        return metrics, threshold

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
