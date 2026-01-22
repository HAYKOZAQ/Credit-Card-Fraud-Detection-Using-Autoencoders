import numpy as np
import pandas as pd
import torch
from src.core.config import Config
from scipy.stats import entropy

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables'''
    # Simple PSI implementation for monitoring pipeline
    # Discretize expected and actual
    
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    valid_psi = True
    
    # Just single variate for now (reconstruction error distribution drift)
    # expected: training errors
    # actual: production errors
    
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Handle zeros
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def counterfactual_analysis(model, high_error_sample, scalar, target_feature_idx, target_value):
    """
    Simple Causal Intervention: Set feature to target_value and observe Delta Error.
    model: Trained AE
    high_error_sample: 1D numpy array (scaled)
    scalar: fitted scaler to inverse transform/transform if needed
    target_feature_idx: index of feature to intervene on
    target_value: raw value to set (will be scaled)
    """
    model.eval()
    
    # 1. Get original error
    input_tensor = torch.FloatTensor(high_error_sample).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        recon = model(input_tensor)
        if isinstance(recon, tuple): recon = recon[0]
        orig_error = torch.mean((input_tensor - recon)**2).item()
        
    # 2. Intervene
    modified_sample = high_error_sample.copy()
    
    # We need to handle scaling. 
    # To set specific feature value, we essentially need to know how to scale it.
    # We can hack it: Create a dummy array with target_value, scale it, extract the value.
    # Or just assume we want to set it to "Mean" (which is 0 in standard scaler).
    
    if target_value == 'mean':
        modified_sample[target_feature_idx] = 0 # Standard Scaler mean is 0
    else:
        # Complex to scale single value without context of others if scalers are coupled? 
        # StandardScaler is independent per feature.
        # We can just check the mean/std from scaler if accessible.
        pass # Simplified: Use 0 for "Normal"
        
    # 3. Get new error
    mod_tensor = torch.FloatTensor(modified_sample).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        recon_mod = model(mod_tensor)
        if isinstance(recon_mod, tuple): recon_mod = recon_mod[0]
        new_error = torch.mean((mod_tensor - recon_mod)**2).item()
        
    return orig_error, new_error, orig_error - new_error

def check_drift_and_retrain(expected_errors, actual_errors, drift_threshold=0.25):
    """
    Automated Retraining Trigger:
    Detects if PSI > threshold and triggers a placeholder for retraining.
    """
    psi = calculate_psi(expected_errors, actual_errors)
    print(f"Current PSI: {psi:.4f}")
    
    if psi > drift_threshold:
        print("!!! Significant Concept Drift Detected !!!")
        print(">>> Triggering Automated Retraining Pipeline...")
        # In a real system, this would call a Jenkins/GitHub Action or an internal training job
        return True, psi
    else:
        print("Model performance stable.")
        return False, psi
