import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import time
from src.core.config import Config

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return MSE + Config.BETA * KLD

def nt_xent_loss(z, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss for Contrastive Learning.
    """
    z = nn.functional.normalize(z, dim=1)
    batch_size = z.shape[0]
    similarity_matrix = torch.matmul(z, z.T)
    
    # Masks for positive pairs (we'll just use the diagonal for self-similarity if no augmentations)
    # Ideally we'd have two views, but for simplicity on tabular, we can use a small perturbation.
    mask = torch.eye(batch_size, device=z.device).bool()
    similarity_matrix = similarity_matrix / temperature
    
    # We want to pull together similar samples. If no labels, this is hard.
    # Usually we use perturbations. Let's assume z is a batch of (z1, z2)
    # For now, a very simple version:
    exp_sim = torch.exp(similarity_matrix)
    return -torch.log(exp_sim.diag() / exp_sim.sum(dim=1)).mean()

def log_experiment(model_name, history, metrics=None):
    log_path = os.path.join(Config.RESULTS_DIR, "experiments.json")
    entry = {
        "timestamp": time.ctime(),
        "model": model_name,
        "epochs": len(history['train_loss']),
        "final_val_loss": history['val_loss'][-1],
        "metrics": metrics
    }
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                 logs = json.load(f)
            except: pass
    logs.append(entry)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=4)

def train_model(model, train_loader, val_loader, epochs=None, adv_train=False):
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    model_name = model.__class__.__name__
    is_vae = model_name == 'VariationalAutoencoder'
    is_contrastive = model_name == 'ContrastiveAutoencoder'
    is_graph = model_name == 'GraphAutoencoder'
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    
    print(f"\nStarting Training for {model_name}...")
    if adv_train: print(">>> Adversarial Training Enabled")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            # Adversarial Perturbation (Simple FGSM for training)
            if adv_train:
                inputs.requires_grad = True
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss_clean = criterion(outputs, targets)
                model.zero_grad()
                loss_clean.backward(retain_graph=True)
                data_grad = inputs.grad.data
                perturbed_inputs = inputs + 0.02 * data_grad.sign()
                inputs = perturbed_inputs.detach()
            
            optimizer.zero_grad()
            
            if is_vae:
                recon_x, mu, logvar = model(inputs)
                loss = vae_loss_function(recon_x, targets, mu, logvar)
            elif is_contrastive:
                recon_x, z = model(inputs)
                loss_recon = criterion(recon_x, targets)
                loss_cont = nt_xent_loss(z)
                loss = loss_recon + 0.1 * loss_cont
            elif is_graph:
                # Identity matrix as simple adj for tabular batch
                adj = torch.eye(inputs.size(0), device=Config.DEVICE)
                recon_x, _ = model(inputs, adj)
                loss = criterion(recon_x, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                if is_vae:
                    recon_x, mu, logvar = model(inputs)
                    loss = vae_loss_function(recon_x, targets, mu, logvar)
                elif is_contrastive:
                    recon_x, z = model(inputs)
                    loss = criterion(recon_x, targets) + 0.1 * nt_xent_loss(z)
                elif is_graph:
                    adj = torch.eye(inputs.size(0), device=Config.DEVICE)
                    recon_x, _ = model(inputs, adj)
                    loss = criterion(recon_x, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name.lower()}.pth")
            torch.save(model.state_dict(), model_save_path)
                
    log_experiment(model_name, history)
    return history
