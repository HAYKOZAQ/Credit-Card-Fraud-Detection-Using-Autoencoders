import numpy as np
import torch
import pickle
from src.core.config import Config

def calculate_psi(expected, actual, buckets=10):
    '''Calculate the PSI (population stability index) across all variables'''
    
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def counterfactual_analysis(model, high_error_sample, target_feature_idx, target_value):
    """
    Simple Causal Intervention: Set feature to target_value and observe Delta Error.
    model: Trained AE
    high_error_sample: 1D numpy array (scaled)
    target_feature_idx: index of feature to intervene on
    target_value: 'mean' or a raw numeric value to set (will be scaled)
    """
    model.eval()
    
    input_tensor = torch.FloatTensor(high_error_sample).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        if model.__class__.__name__ == 'GraphAutoencoder':
            adj = torch.eye(1).to(Config.DEVICE)
            recon = model(input_tensor, adj)
        else:
            recon = model(input_tensor)
        if isinstance(recon, tuple): recon = recon[0]
        orig_error = torch.mean((input_tensor - recon)**2).item()
        
    modified_sample = high_error_sample.copy()
    
    if target_value == 'mean':
        modified_sample[target_feature_idx] = 0
    else:
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        scaled_val = (target_value - scaler.mean_[target_feature_idx]) / scaler.scale_[target_feature_idx]
        modified_sample[target_feature_idx] = scaled_val
        
    # 3. Get new error
    mod_tensor = torch.FloatTensor(modified_sample).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        if model.__class__.__name__ == 'GraphAutoencoder':
            adj = torch.eye(1).to(Config.DEVICE)
            recon_mod = model(mod_tensor, adj)
        else:
            recon_mod = model(mod_tensor)
        if isinstance(recon_mod, tuple): recon_mod = recon_mod[0]
        new_error = torch.mean((mod_tensor - recon_mod)**2).item()
        
    return orig_error, new_error, orig_error - new_error
