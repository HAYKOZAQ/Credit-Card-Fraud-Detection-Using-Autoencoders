import torch
import torch.nn as nn
import numpy as np
from src.core.config import Config
from src.evaluation.evaluate import get_reconstruction_errors
import torch.utils.data

def generate_adversarial_examples(model, data_loader, epsilon=0.1, mode='maximize'):
    """
    Generate adversarial examples using FGSM-like approach.
    mode='maximize': perturbs to MAXIMIZE reconstruction error (sensitivity test).
    mode='minimize': perturbs to MINIMIZE reconstruction error (evasion attack simulation).
    """
    model.eval()
    adversarial_examples = []
    original_data = []
    
    criterion = nn.MSELoss()
    
    for inputs, _ in data_loader:
        inputs = inputs.to(Config.DEVICE)
        inputs.requires_grad = True
        
        if model.__class__.__name__ == 'GraphAutoencoder':
            n_nodes = inputs.shape[0]
            edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).to(Config.DEVICE)
            outputs = model(inputs, edge_index)
        else:
            outputs = model(inputs)
        if isinstance(outputs, tuple): outputs = outputs[0]
        
        loss = criterion(outputs, inputs)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        if mode == 'maximize':
            perturbed_inputs = inputs + epsilon * data_grad.sign()
        else:
            perturbed_inputs = inputs - epsilon * data_grad.sign()
        
        adversarial_examples.append(perturbed_inputs.detach().cpu())
        original_data.append(inputs.detach().cpu())
        
    return torch.cat(adversarial_examples), torch.cat(original_data)

def evaluate_robustness(models, test_loader, epsilon=0.1):
    """
    Compare models on clean and adversarial data.
    Returns both sensitivity (maximize error) and evasion resilience.
    models: dict {'name': model}
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating robustness for {name}...")
        adv_inputs, orig_inputs = generate_adversarial_examples(model, test_loader, epsilon, mode='maximize')
        evasion_inputs, _ = generate_adversarial_examples(model, test_loader, epsilon, mode='minimize')
        
        adv_dataset = torch.utils.data.TensorDataset(adv_inputs, adv_inputs)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=Config.BATCH_SIZE)
        
        eva_dataset = torch.utils.data.TensorDataset(evasion_inputs, evasion_inputs)
        eva_loader = torch.utils.data.DataLoader(eva_dataset, batch_size=Config.BATCH_SIZE)
        
        clean_errors = get_reconstruction_errors(model, test_loader)
        adv_errors = get_reconstruction_errors(model, adv_loader)
        evasion_errors = get_reconstruction_errors(model, eva_loader)
        
        results[name] = {
            'Clean Mean Error': np.mean(clean_errors),
            'Adversarial Mean Error (Max)': np.mean(adv_errors),
            'Evasion Mean Error (Min)': np.mean(evasion_errors),
            'Sensitivity (Max/Clean)': np.mean(adv_errors) / (np.mean(clean_errors) + 1e-9),
            'Evasion Resistance (Min/Clean)': np.mean(evasion_errors) / (np.mean(clean_errors) + 1e-9)
        }
        
    return results
