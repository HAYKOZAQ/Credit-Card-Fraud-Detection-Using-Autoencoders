"""
Model Monitoring Module
Tracks model performance, drift, and alerts in production
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from dataclasses import dataclass, asdict
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    timestamp: str
    is_fraud: bool
    fraud_probability: float
    reconstruction_error: float
    threshold: float
    features: Dict[str, float]
    actual_label: Optional[bool] = None  # Filled in later when ground truth available


@dataclass
class DriftMetrics:
    """Metrics for detecting data drift"""
    timestamp: str
    psi_score: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    feature_drift: Dict[str, float]  # Per-feature drift scores
    alert_triggered: bool


@dataclass
class PerformanceMetrics:
    """Model performance metrics over a time window"""
    timestamp: str
    window_hours: int
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    avg_reconstruction_error: float
    avg_fraud_probability: float
    # If ground truth available
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(
        self,
        log_dir: str = "logs/monitoring",
        window_size: int = 1000,  # Number of predictions to keep in memory
        alert_threshold_psi: float = 0.2,  # PSI threshold for drift alert
        alert_threshold_fraud_rate: float = 0.1,  # Fraud rate change threshold
        on_drift_detected_callback = None
    ):
        self.log_dir = log_dir
        self.window_size = window_size
        self.alert_threshold_psi = alert_threshold_psi
        self.alert_threshold_fraud_rate = alert_threshold_fraud_rate
        self.on_drift_detected_callback = on_drift_detected_callback
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # In-memory storage for recent predictions
        self.predictions = deque(maxlen=window_size)
        
        # Reference distribution for drift detection (from training data)
        self.reference_distribution = None
        
        logger.info(f"ModelMonitor initialized with window_size={window_size}")
    
    def log_prediction(self, record: PredictionRecord):
        """Log a single prediction"""
        self.predictions.append(record)
        
        # Periodically save to disk
        if len(self.predictions) % 100 == 0:
            self._save_predictions_to_disk()
    
    def _save_predictions_to_disk(self):
        """Save predictions to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.log_dir, f"predictions_{timestamp}.json")
            
            records = [asdict(p) for p in self.predictions]
            with open(filepath, 'w') as f:
                json.dump(records, f, indent=2)
            
            logger.info(f"Saved {len(records)} predictions to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    
    def set_reference_distribution(self, data: pd.DataFrame, features: List[str]):
        """Set reference distribution for drift detection"""
        self.reference_distribution = {}
        for feature in features:
            if feature in data.columns:
                self.reference_distribution[feature] = {
                    'mean': float(data[feature].mean()),
                    'std': float(data[feature].std()),
                    'histogram': np.histogram(data[feature], bins=10, density=True)[0].tolist()
                }
        logger.info(f"Reference distribution set for {len(features)} features")
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        breakpoints = np.percentile(expected, np.arange(0, buckets + 1) / buckets * 100)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Replace zeros with small value to avoid log(0)
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)
    
    def detect_drift(self, features: List[str]) -> Optional[DriftMetrics]:
        """Detect data drift in recent predictions"""
        if not self.reference_distribution or len(self.predictions) < 100:
            return None
        
        try:
            # Extract feature values from recent predictions
            recent_data = []
            for pred in self.predictions:
                recent_data.append(pred.features)
            
            df_recent = pd.DataFrame(recent_data)
            
            # Calculate drift for each feature
            feature_drift = {}
            psi_scores = []
            ks_stats = []
            
            for feature in features:
                if feature in df_recent.columns and feature in self.reference_distribution:
                    # Get recent values
                    recent_values = df_recent[feature].dropna().values
                    
                    if len(recent_values) < 50:
                        continue
                    
                    # Calculate PSI
                    ref_mean = self.reference_distribution[feature]['mean']
                    ref_std = self.reference_distribution[feature]['std']
                    
                    # Generate reference samples (approximation)
                    ref_samples = np.random.normal(ref_mean, ref_std, len(recent_values))
                    
                    psi = self.calculate_psi(ref_samples, recent_values)
                    feature_drift[feature] = psi
                    psi_scores.append(psi)
                    
                    # Simple KS statistic approximation
                    ks = abs(np.mean(recent_values) - ref_mean) / (ref_std + 1e-6)
                    ks_stats.append(ks)
            
            if not psi_scores:
                return None
            
            avg_psi = np.mean(psi_scores)
            avg_ks = np.mean(ks_stats)
            alert_triggered = avg_psi > self.alert_threshold_psi
            
            metrics = DriftMetrics(
                timestamp=datetime.now().isoformat(),
                psi_score=avg_psi,
                ks_statistic=avg_ks,
                feature_drift=feature_drift,
                alert_triggered=alert_triggered
            )
            
            if alert_triggered:
                logger.warning(f"DRIFT ALERT: PSI={avg_psi:.4f} > threshold={self.alert_threshold_psi}")
                self._save_drift_alert(metrics)
                if self.on_drift_detected_callback is not None:
                    try:
                        self.on_drift_detected_callback()
                    except Exception as e:
                        logger.error(f"Failed to execute drift callback: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return None
    
    def _save_drift_alert(self, metrics: DriftMetrics):
        """Save drift alert to disk"""
        try:
            filepath = os.path.join(self.log_dir, "drift_alerts.json")
            
            alerts = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    alerts = json.load(f)
            
            alerts.append(asdict(metrics))
            
            with open(filepath, 'w') as f:
                json.dump(alerts, f, indent=2)
            
            logger.info(f"Drift alert saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save drift alert: {e}")
    
    def get_performance_metrics(self, window_hours: int = 24) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics over a time window"""
        if len(self.predictions) == 0:
            return None
        
        try:
            # Filter predictions within time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            recent_preds = [
                p for p in self.predictions
                if datetime.fromisoformat(p.timestamp) > cutoff_time
            ]
            
            if len(recent_preds) == 0:
                return None
            
            # Calculate metrics
            total = len(recent_preds)
            fraud_preds = [p for p in recent_preds if p.is_fraud]
            fraud_count = len(fraud_preds)
            
            avg_error = np.mean([p.reconstruction_error for p in recent_preds])
            avg_prob = np.mean([p.fraud_probability for p in recent_preds])
            
            # Check if we have ground truth
            with_truth = [p for p in recent_preds if p.actual_label is not None]
            
            precision = recall = f1 = None
            if len(with_truth) > 10:
                tp = sum(1 for p in with_truth if p.is_fraud and p.actual_label)
                fp = sum(1 for p in with_truth if p.is_fraud and not p.actual_label)
                fn = sum(1 for p in with_truth if not p.is_fraud and p.actual_label)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                window_hours=window_hours,
                total_predictions=total,
                fraud_predictions=fraud_count,
                fraud_rate=fraud_count / total,
                avg_reconstruction_error=float(avg_error),
                avg_fraud_probability=float(avg_prob),
                precision=precision,
                recall=recall,
                f1_score=f1
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return None
    
    def get_summary(self) -> Dict:
        """Get monitoring summary"""
        return {
            "total_predictions": len(self.predictions),
            "window_size": self.window_size,
            "last_prediction": self.predictions[-1].timestamp if self.predictions else None,
            "reference_distribution_set": self.reference_distribution is not None,
            "log_dir": self.log_dir
        }


def trigger_retraining():
    """
    Trigger automatic retraining of the model.
    Loads the drift simulator and runs retraining.
    """
    logger.info(">>> AUTO-RETRAINING TRIGGERED BY DRIFT DETECTION <<<")
    try:
        from src.evaluation.simulation import DriftSimulator
        sim = DriftSimulator()
        months = sim.simulate_months(num_months=4)
        sim.run_drift_experiment(months)
        logger.info(">>> AUTO-RETRAINING COMPLETED SUCCESSFULLY <<<")
    except Exception as e:
        logger.error(f"Error during auto-retraining: {e}")


# Global monitor instance
_monitor_instance = None


def get_monitor() -> ModelMonitor:
    """Get or create global monitor instance with auto-retraining callback"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor(on_drift_detected_callback=trigger_retraining)
    return _monitor_instance
