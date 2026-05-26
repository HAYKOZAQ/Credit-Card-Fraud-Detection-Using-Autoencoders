import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.core.config import Config


def get_reconstruction_errors(model, dataloader):
    model.eval()
    errors = []
    model_name = model.__class__.__name__
    is_vae = model_name == 'VariationalAutoencoder'
    is_contrastive = model_name == 'ContrastiveAutoencoder'
    is_graph = model_name == 'GraphAutoencoder'
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(Config.DEVICE)
            if is_vae:
                recon_x, _, _ = model(inputs)
            elif is_contrastive:
                recon_x, _ = model(inputs)
            elif is_graph:
                dists = torch.cdist(inputs, inputs)
                sigma = dists.mean()
                adj = torch.exp(-dists / (sigma + 1e-8))
                adj = adj / adj.sum(dim=1, keepdim=True)
                recon_x, _ = model(inputs, adj)
            else:
                recon_x = model(inputs)
                
            if len(inputs.shape) == 3: # Sequential
                mse = torch.mean((inputs - recon_x)**2, dim=(1, 2))
            else: # Standard
                mse = torch.mean((inputs - recon_x)**2, dim=1)
            errors.extend(mse.cpu().numpy())
    return np.array(errors)

def plot_history(history, model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    img_path = os.path.join(Config.VIZ_DIR, f"training_history_{model_name}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Training history plot saved to: {img_path}")

def plot_feature_importance(model, fraud_loader, feature_names):
    model.eval()
    all_inputs = []
    all_reconstructions = []
    is_vae = model.__class__.__name__ == 'VariationalAutoencoder'
    
    # Sequential models don't easily map to 1D feature importance without attention or averaging
    if model.__class__.__name__ == 'LSTMAutoencoder':
        print("Feature importance skipped for LSTM (sequential architecture).")
        return

    with torch.no_grad():
        for inputs, _ in fraud_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            if is_vae:
                recon_x, _, _ = outputs
            else:
                recon_x = outputs
                
            all_inputs.append(inputs.cpu().numpy())
            all_reconstructions.append(recon_x.cpu().numpy())
            
    X_fraud = np.concatenate(all_inputs)
    X_recon = np.concatenate(all_reconstructions)
    
    # Feature-wise MSE
    mse_per_feature = np.mean((X_fraud - X_recon)**2, axis=0)
    
    plt.figure(figsize=(10, 8))
    # Sort features for visualization
    indices = np.argsort(mse_per_feature)
    plt.barh(np.array(feature_names)[indices], mse_per_feature[indices], color='skyblue')
    plt.title("Global Feature Importance")
    plt.xlabel("Mean Reconstruction Error")
    plt.tight_layout()
    img_path = os.path.join(Config.VIZ_DIR, f"feature_importance_{model.__class__.__name__.lower()}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Feature importance saved to: {img_path}")

def get_uncertainty_scores(model, dataloader, num_passes=5):
    """
    Computes uncertainty using MC Dropout.
    Returns mean reconstruction error and variance (uncertainty).
    """
    model.train() # Enable dropout
    all_errors = []
    all_variances = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(Config.DEVICE)
            batch_outputs = []
            
            for _ in range(num_passes):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                batch_outputs.append(outputs.unsqueeze(0))
            
            # [passes, batch, feats]
            stacked = torch.cat(batch_outputs, dim=0)
            
            # Mean reconstruction across passes
            mean_recon = torch.mean(stacked, dim=0)
            
            # Error of the mean reconstruction
            if len(inputs.shape) == 3:
                mse = torch.mean((inputs - mean_recon)**2, dim=(1, 2))
            else:
                mse = torch.mean((inputs - mean_recon)**2, dim=1)
                
            # Variance of the reconstruction (uncertainty)
            variance = torch.var(stacked, dim=0).mean(dim=-1)
            if len(inputs.shape) == 3:
                 variance = variance.mean(dim=-1)
            
            all_errors.extend(mse.cpu().numpy())
            all_variances.extend(variance.cpu().numpy())
            
    model.eval()
    return np.array(all_errors), np.array(all_variances)

def plot_latent_tsne(model, test_loader, fraud_loader):
    model.eval()
    latents = []
    labels = []
    
    def extract(loader, label):
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(Config.DEVICE)
                if hasattr(model, 'encoder'):
                    x = model.encoder(inputs)
                    if isinstance(x, tuple):
                        x = x[0]
                    if len(x.shape) > 2:
                        x = x.mean(dim=1)
                    latents.append(x.cpu().numpy())
                    labels.extend([label]*inputs.size(0))
                elif hasattr(model, 'encode'): # VAE
                    mu, _ = model.encode(inputs)
                    latents.append(mu.cpu().numpy())
                    labels.extend([label]*inputs.size(0))

    if hasattr(model, 'encoder') or hasattr(model, 'encode'):
        extract(test_loader, 0)
        extract(fraud_loader, 1)
    else:
        print("Skipping TSNE: Model structure not supported")
        return

    X = np.concatenate(latents)
    y = np.array(labels)
    
    # Subsample
    if len(X) > 2000:
        idx = np.random.choice(len(X), 2000, replace=False)
        X = X[idx]
        y = y[idx]
        
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_tsne['Label'] = ['Fraud' if i==1 else 'Normal' for i in y]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='Label', palette={'Fraud': 'red', 'Normal': 'blue'}, alpha=0.6)
    plt.title(f't-SNE of Latent Space ({model.__class__.__name__})')
    
    viz_path = os.path.join(Config.VIZ_DIR, f"tsne_{model.__class__.__name__}.png")
    plt.savefig(viz_path)
    plt.close()
    print(f"TSNE plot saved to {viz_path}")

def plot_uncertainty_vs_error(errors, uncertainties, labels):
    df = pd.DataFrame({
        'Reconstruction Error': errors,
        'Uncertainty': uncertainties,
        'Type': ['Fraud' if l==1 else 'Normal' for l in labels]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Reconstruction Error', y='Uncertainty', hue='Type', palette={'Fraud': 'red', 'Normal': 'blue'}, alpha=0.6)
    plt.title('Uncertainty vs Reconstruction Error')
    
    viz_path = os.path.join(Config.VIZ_DIR, "uncertainty_vs_error.png")
    plt.savefig(viz_path)
    plt.close()
    print(f"Uncertainty plot saved to {viz_path}")

def plot_merchant_heatmap(fraud_df, reconstruction_errors):
    df = fraud_df.copy()
    
    if len(df) != len(reconstruction_errors):
        min_len = min(len(df), len(reconstruction_errors))
        print(f"Warning: Mismatched lengths — df={len(df)}, errors={len(reconstruction_errors)}. Truncating to {min_len}.")
        df = df.iloc[:min_len]
        reconstruction_errors = reconstruction_errors[:min_len]
        
    df['error'] = reconstruction_errors
    
    if 'merchant' not in df.columns:
        print("Warning: 'merchant' column missing — plots without merchant group-by.")
        df['merchant'] = 'unknown'

    merch_stats = df.groupby('merchant').agg(error=('error', 'mean'), volume=('error', 'count')).reset_index()
    merch_stats = merch_stats[merch_stats['volume'] > 5]
    if 'is_fraud' in df.columns:
        fraud_rate = df.groupby('merchant')['is_fraud'].mean().reset_index()
        merch_stats = merch_stats.merge(fraud_rate, on='merchant')
    
    plt.figure(figsize=(12, 8))
    y_col = 'is_fraud' if 'is_fraud' in merch_stats.columns else 'error'
    plt.hist2d(merch_stats['error'], merch_stats[y_col], bins=20, cmap='viridis')
    plt.colorbar(label='Count of Merchants')
    plt.title('Merchant Landscape: Fraud Rate vs Reconstruction Error')
    plt.xlabel('Avg Reconstruction Error')
    plt.ylabel('Fraud Rate')
    
    viz_path = os.path.join(Config.VIZ_DIR, "merchant_heatmap.png")
    plt.savefig(viz_path)
    plt.close()
    print(f"Merchant heatmap saved to {viz_path}")

def plot_cumulative_fraud(model, test_df, fraud_df, test_loader, fraud_loader,
                          normal_errors=None, fraud_errors=None):
    model.eval()
    if normal_errors is None:
        normal_errors = get_reconstruction_errors(model, test_loader)
    if fraud_errors is None:
        fraud_errors = get_reconstruction_errors(model, fraud_loader)
    
    test_res = test_df.copy()
    test_res['error'] = normal_errors
    test_res['label'] = 0
    
    fraud_res = fraud_df.copy()
    fraud_res['error'] = fraud_errors
    fraud_res['label'] = 1
    
    combined = pd.concat([test_res, fraud_res])
    time_col = 'trans_date_trans_time' if 'trans_date_trans_time' in combined.columns else None
    if time_col and combined[time_col].dtype != 'datetime64[ns]':
        combined[time_col] = pd.to_datetime(combined[time_col], errors='coerce')
    
    if time_col and not combined[time_col].isna().all():
        combined = combined.sort_values(time_col)
        threshold = np.mean(normal_errors) + 3*np.std(normal_errors)
        combined['predicted_fraud'] = combined['error'] > threshold
        combined['cumulative_frauds_detected'] = combined['predicted_fraud'].cumsum()
        combined['cumulative_actual_frauds'] = combined['label'].cumsum()
        plt.figure(figsize=(12, 6))
        plt.plot(combined[time_col], combined['cumulative_frauds_detected'], label='Detected Frauds')
        plt.plot(combined[time_col], combined['cumulative_actual_frauds'], label='Actual Frauds')
        plt.title('Cumulative Fraud Detection Over Time')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        viz_path = os.path.join(Config.VIZ_DIR, "cumulative_fraud.png")
        plt.savefig(viz_path)
        plt.close()
        print(f"Cumulative plot saved to {viz_path}")
    else:
        combined = combined.reset_index(drop=True)
        combined['sample_index'] = combined.index
        threshold = np.mean(normal_errors) + 3*np.std(normal_errors)
        combined['predicted_fraud'] = combined['error'] > threshold
        combined['cumulative_frauds_detected'] = combined['predicted_fraud'].cumsum()
        combined['cumulative_actual_frauds'] = combined['label'].cumsum()
        plt.figure(figsize=(12, 6))
        plt.plot(combined['sample_index'], combined['cumulative_frauds_detected'], label='Detected Frauds')
        plt.plot(combined['sample_index'], combined['cumulative_actual_frauds'], label='Actual Frauds')
        plt.title('Cumulative Fraud Detection Over Samples')
        plt.xlabel('Sample Index')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        viz_path = os.path.join(Config.VIZ_DIR, "cumulative_fraud.png")
        plt.savefig(viz_path)
        plt.close()
        print(f"Cumulative plot saved to {viz_path}")
def calculate_shap_values(model, train_loader, test_loader, feature_names):
    """
    Computes SHAP values for the Autoencoder.
    """
    model.eval()
    # Use a small background dataset from train_loader
    background_data = []
    for inputs, _ in train_loader:
        background_data.append(inputs)
        if len(background_data) > 5: break
    background_data = torch.cat(background_data, dim=0)[:100].to(Config.DEVICE)
    
    # Wrapper function for SHAP (outputs MSE reconstruction error)
    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, tuple): outputs = outputs[0]
            mse = torch.mean((x_tensor - outputs)**2, dim=1)
        return mse.cpu().numpy()

    explainer = shap.KernelExplainer(model_wrapper, background_data.cpu().numpy())
    
    test_samples = []
    for inputs, _ in test_loader:
        test_samples.append(inputs)
        if len(test_samples) > 1: break
    test_samples = torch.cat(test_samples, dim=0)[:10].cpu().numpy()
    
    try:
        shap_values = explainer.shap_values(test_samples)
    except Exception as e:
        print(f"Warning: SHAP computation failed ({e}). Skipping SHAP summary plot.")
        return None
    
    plt.figure()
    shap.summary_plot(shap_values, test_samples, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Attribution (Reconstruction Error)")
    img_path = os.path.join(Config.VIZ_DIR, "shap_summary.png")
    plt.savefig(img_path)
    plt.close()
    print(f"SHAP summary saved to {img_path}")
    return shap_values

def ensemble_scoring(models_dict, dataloader):
    """
    Weights the reconstruction errors of multiple models.
    """
    all_scores = []
    weights = []
    
    for name, model in models_dict.items():
        errors = get_reconstruction_errors(model, dataloader)
        # Normalize errors to [0, 1] range for fair weighting?
        # Or just use raw MSE. Better to normalize by max.
        norm_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        if np.std(errors) < 1e-10:
            norm_errors = np.zeros_like(errors)
        else:
            norm_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        all_scores.append(norm_errors)
        
        if 'vae' in name.lower() or 'attention' in name.lower():
            weights.append(1.5)
        else:
            weights.append(1.0)
            
    all_scores = np.array(all_scores)
    weights = np.array(weights).reshape(-1, 1)
    ensemble_score = (all_scores * weights).sum(axis=0) / weights.sum()
    
    return ensemble_score
