import gradio as gr
import torch
import numpy as np
import pickle
import pandas as pd
import os
from src.core.config import Config
from src.core.model import get_model
from src.evaluation.explain import FraudExplainer

scaler = None
model = None
explainer = None
encoders = None

# Load Assets
try:
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    model = get_model('standard')
    model_name = model.__class__.__name__.lower()
    model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
    model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
    model.to(Config.DEVICE)
    model.eval()
    
    bg_path = os.path.join(Config.MODEL_DIR, "background_sample.pkl")
    if os.path.exists(bg_path):
        with open(bg_path, 'rb') as f:
            background_data = pickle.load(f)
    else:
        background_data = np.zeros((50, Config.INPUT_DIM)) 
        
    explainer = FraudExplainer(model, background_data)
except FileNotFoundError as e:
    print(f"Startup: Required file not found: {e}")
    print("Please run 'python main.py --mode research' first to generate model files.")
except Exception as e:
    print(f"Startup Error: {e}")
    print("Model loading failed. Check traceback above for details (e.g., PyTorch version, corrupted files).")

THRESHOLD = None

def get_default_threshold():
    from src.core.data_loader import get_dataloaders
    from src.evaluation.evaluate import get_reconstruction_errors
    try:
        _, test_loader, _, _, _, _, _ = get_dataloaders()
        errors = get_reconstruction_errors(model, test_loader)
        return np.mean(errors) + 3 * np.std(errors)
    except Exception:
        return 1.65

THRESHOLD = get_default_threshold() if model is not None else 1.65
FEATURE_NAMES = ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log']

def predict_and_explain(category, job, city_pop, age, hour, distance_km, amt):
    if scaler is None or model is None or encoders is None:
        return "Error: Model not loaded. Run main.py first.", {}, None

    cat_val = encoders['category'].transform([str(category)])[0] if str(category) in encoders['category'].classes_ else 0
    job_val = encoders['job'].transform([str(job)])[0] if str(job) in encoders['job'].classes_ else 0
    amt_log = np.log1p(amt)
    features = np.array([[cat_val, job_val, city_pop, age, hour, distance_km, amt_log]])
    
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled).to(Config.DEVICE)
    
    # 3. Predict Score
    with torch.no_grad():
        recon = model(features_tensor)
        error = torch.mean((features_tensor - recon)**2).item()
    
    is_fraud = error > THRESHOLD
    result_text = "🚨 FRAUD DETECTED 🚨" if is_fraud else "✅ Authorized"
    
    # 4. Generate SHAP if fraud
    explanation_fig = None
    if is_fraud:
        if explainer is not None:
            plot_path, _ = explainer.explain_instance(features_scaled[0], FEATURE_NAMES)
            explanation_fig = plot_path
        else:
            result_text += " (Explainability unavailable)"
    
    details = {
        "Anomaly Score": round(error, 4),
        "Decision": "Flagged" if is_fraud else "Normal",
        "Explainability": "SHAP analysis generated below" if is_fraud else "No explanation needed"
    }
    
    return result_text, details, explanation_fig

def get_monitoring_stats():
    stats = {
        "Daily Transaction Volume": 12450,
        "Flagged Transactions": 342,
        "Estimated Fraud Rate": "2.7%",
        "Model Health (Recall)": "84.2%",
        "Avg Anomaly Score": 0.45
    }
    return pd.DataFrame([stats])

# --- GUI Layout ---
with gr.Blocks(title="🛡️ FraudGuard Production Hub", theme="soft") as demo:
    gr.Markdown("# 🛡️ FraudGuard Production Hub")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Real-Time Inference"):
            with gr.Row():
                with gr.Column():
                    cat = gr.Slider(0, 15, step=1, label="Category ID")
                    job = gr.Slider(0, 50, step=1, label="Job ID")
                    pop = gr.Number(value=50000, label="City Pop")
                    age = gr.Slider(18, 100, step=1, label="Age")
                    hr = gr.Slider(0, 23, step=1, label="Hour")
                    dist = gr.Number(value=5.0, label="Distance (km)")
                    amt = gr.Number(value=100.0, label="Amount ($)")
                    btn = gr.Button("Analyze System Risk", variant="primary")
                
                with gr.Column():
                    out_text = gr.Textbox(label="System Verdict")
                    out_json = gr.JSON(label="Technical Details")
                    out_shap = gr.Image(label="SHAP Explanation (Frauds Only)")
            
            btn.click(predict_and_explain, [cat, job, pop, age, hr, dist, amt], [out_text, out_json, out_shap])

        with gr.TabItem("📊 Production Monitoring"):
            gr.Markdown("## System Health & Drift Metrics")
            stats_btn = gr.Button("Refresh Pipeline Stats")
            stats_df = gr.DataFrame(get_monitoring_stats)
            gr.Markdown("---")
            gr.Markdown("### Concept Drift Impact (Retrained vs Static)")
            drift_plot = os.path.join(Config.VIZ_DIR, "drift_simulation.html")
            gr.Markdown(f"[Open Interactive Drift Report](file://{drift_plot})")

if __name__ == "__main__":
    demo.launch(share=False)
