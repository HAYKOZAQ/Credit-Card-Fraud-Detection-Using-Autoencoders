import torch
import torch.nn as nn
import numpy as np
from src.core.config import Config
from src.evaluation.evaluate import get_reconstruction_errors
import torch.utils.data

def generate_adversarial_examples(model, data_loader, epsilon=0.1):
    """
    Generate adversarial examples using FGSM-like approach.
    Since we want to MAXIMIZE reconstruction error (to simulate evasion or just test sensitivity),
    usually attackers want to MINIMIZE error to hide fraud (evasion).
    BUT here, "Fraudsters try to evade detection".
    If our detection is based on HIGH reconstruction error => Fraudsters want LOW reconstruction error.
    So they would perturb input to MINIMIZE reconstruction error.
    
    However, the prompt says: "Generate adversarial examples by perturbing transaction features... and see how much reconstruction error changes".
    And "Train a denoising autoencoder that reconstructs well even on noisy/perturbed inputs".
    
    If we want to test robustness to NOISE (perturbations), we usually just add random noise or adversarial noise that MAXIMIZES error to see if model breaks?
    Actually, Denoising AE is robust to noise because it learns to remove it.
    If we want to test "Adversarial Robustness" in anomaly detection:
    1. Attack: Fraudster modifies fraud sample to look normal (Low Recon Error).
    2. Resilience: Model should still give High Recon Error for fraud even if perturbed.
    
    The prompt says: "see how much reconstruction error changes".
    
    Let's implement an attack that tries to REDUCE reconstruction error (Simulating fraudster hiding).
    And also maybe random noise sensitivity.
    
    Let's stick to the prompt: "Generate adversarial examples by perturbing ... and see how much reconstruction error changes".
    I will implement an attack that tries to maximize error (sensitivity) AND minimize error (evasion).
    But standard "Robustness" often implies stability.
    I'll implement FGSM to MAXIMIZE error (standard adversarial example generation for classification, but here regression).
    Actually, let's implement Gradient Ascent on Loss => Maximize Error.
    """
    model.eval()
    adversarial_examples = []
    original_data = []
    
    criterion = nn.MSELoss()
    
    for inputs, _ in data_loader:
        inputs = inputs.to(Config.DEVICE)
        inputs.requires_grad = True
        
        outputs = model(inputs)
        if isinstance(outputs, tuple): outputs = outputs[0]
        
        loss = criterion(outputs, inputs)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        # To MAXIMIZE Error (Test stability/worst case noise): inputs + eps * sign(grad)
        # To MINIMIZE Error (Evasion attack): inputs - eps * sign(grad)
        perturbed_inputs = inputs + epsilon * data_grad.sign()
        
        adversarial_examples.append(perturbed_inputs.detach().cpu())
        original_data.append(inputs.detach().cpu())
        
    return torch.cat(adversarial_examples), torch.cat(original_data)

def evaluate_robustness(models, test_loader, epsilon=0.1):
    """
    Compare models on clean and adversarial data.
    models: dict {'name': model}
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating robustness for {name}...")
        adv_inputs, orig_inputs = generate_adversarial_examples(model, test_loader, epsilon)
        
        # Loader for adv
        adv_dataset = torch.utils.data.TensorDataset(adv_inputs, adv_inputs)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=Config.BATCH_SIZE)
        
        # Errors
        clean_errors = get_reconstruction_errors(model, test_loader)
        adv_errors = get_reconstruction_errors(model, adv_loader)
        
        results[name] = {
            'Clean Mean Error': np.mean(clean_errors),
            'Adversarial Mean Error': np.mean(adv_errors),
            'Sensitivity (Adv/Clean)': np.mean(adv_errors) / (np.mean(clean_errors) + 1e-9)
        }
        
    return results
