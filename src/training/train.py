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

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss for Contrastive Learning.
    z1, z2: two augmented views of the same batch [batch_size, latent_dim].
    Positive pairs: corresponding samples across views (diagonal).
    """
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    batch_size = z1.shape[0]
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * (~mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    pos_pairs = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ], dim=0)
    loss = -log_prob[torch.arange(2 * batch_size), pos_pairs].mean()
    return loss

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
            except (json.JSONDecodeError, ValueError):
                 print(f"Warning: Corrupt experiment log at {log_path}, creating new log.")
    logs.append(entry)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=4)

def _build_adjacency(inputs):
    dists = torch.cdist(inputs, inputs)
    sigma = dists.mean()
    adj = torch.exp(-dists / (sigma + 1e-8))
    adj = adj / adj.sum(dim=1, keepdim=True)
    return adj

def train_model(model, train_loader, val_loader, epochs=None, adv_train=False, save_name=None, lr=None):
    learning_rate = lr if lr is not None else Config.LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model_name = model.__class__.__name__
    is_vae = model_name == 'VariationalAutoencoder'
    is_contrastive = model_name == 'ContrastiveAutoencoder'
    is_graph = model_name == 'GraphAutoencoder'
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    
    print(f"\nStarting Training for {model_name}...")
    if adv_train: print(">>> Adversarial Training Enabled")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            if is_graph:
                batch = batch_data.to(Config.DEVICE)
                inputs = batch.x
                targets = batch.x # Autoencoder reconstructs inputs
                edge_index = batch.edge_index
                batch_size = batch.batch_size if hasattr(batch, 'batch_size') else inputs.size(0)
            else:
                inputs, targets = batch_data
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                batch_size = inputs.size(0)
            
            if adv_train and not is_graph:
                inputs_adv = inputs.clone().detach().requires_grad_(True)
                outputs_adv = model(inputs_adv)
                if isinstance(outputs_adv, tuple): outputs_adv = outputs_adv[0]
                loss_adv = criterion(outputs_adv, targets)
                loss_adv.backward()
                data_grad = inputs_adv.grad.data
                inputs = (inputs + 0.02 * data_grad.sign()).detach()
            
            optimizer.zero_grad()
            
            if is_vae:
                recon_x, mu, logvar = model(inputs)
                loss = vae_loss_function(recon_x, targets, mu, logvar)
            elif is_contrastive:
                inputs_a = inputs + torch.randn_like(inputs) * 0.05
                inputs_b = inputs + torch.randn_like(inputs) * 0.05
                recon_x, z_a = model(inputs_a)
                _, z_b = model(inputs_b)
                loss_recon = criterion(recon_x, targets)
                loss_cont = nt_xent_loss(z_a, z_b)
                loss = loss_recon + 0.1 * loss_cont
            elif is_graph:
                recon_x, _ = model(inputs, edge_index)
                loss = criterion(recon_x, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
        
        train_loss /= len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader.sampler)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if is_graph:
                    batch = batch_data.to(Config.DEVICE)
                    inputs = batch.x
                    targets = batch.x
                    edge_index = batch.edge_index
                    batch_size = batch.batch_size if hasattr(batch, 'batch_size') else inputs.size(0)
                else:
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                    batch_size = inputs.size(0)

                if is_vae:
                    recon_x, mu, logvar = model(inputs)
                    loss = vae_loss_function(recon_x, targets, mu, logvar)
                elif is_contrastive:
                    recon_x, z = model(inputs)
                    loss = criterion(recon_x, targets)
                elif is_graph:
                    recon_x, _ = model(inputs, edge_index)
                    loss = criterion(recon_x, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * batch_size
        
        val_loss /= len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader.sampler)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_filename = save_name if save_name else f"{model_name.lower()}.pth"
            model_save_path = os.path.join(Config.MODEL_DIR, save_filename)
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                break
                
    log_experiment(model_name, history)
    return history
