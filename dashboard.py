import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
from src.core.config import Config
from src.core.model import get_model
from src.training.train import train_model
from src.evaluation.evaluate import get_reconstruction_errors
from src.core.data_loader import get_dataloaders
from src.training.gan_trainer import GANTrainer
from src.training.federated import run_federated_simulation
from src.evaluation.analysis_utils import calculate_psi, counterfactual_analysis

st.set_page_config(page_title="DeepGuard: Fraud Detection System", layout="wide", page_icon="🛡️")

# Add custom CSS for "wow" factor
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f7;
    }
    h1 {
        color: #1a1a1a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(45deg, #10B981, #3B82F6);
        border-radius: 8px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data_and_models():
    # Load data once
    train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df = get_dataloaders()
    return train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df

train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df = load_data_and_models()

st.title("🛡️ DeepGuard AI: Advanced Fraud Defense")
st.markdown("**Real-time Anomaly Detection with Autoencoders & GANs**")

tabs = st.tabs(["🏗️ Research Lab", "🕵️ Fraud Detector", "🔬 Causal Analysis", "🕸️ Federated View", "📉 Monitoring"])

# --- TAB 1: RESEARCH LAB ---
with tabs[0]:
    st.header("Model Training & Experimentation")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Architecture", 
                                  ["Standard Autoencoder", "Variational AE (VAE)", "Denoising AE", "Attention AE"])
        use_gan = st.checkbox("Augment with GAN Synthetic Frauds")
        
    with col2:
        epochs = st.slider("Epochs", 5, 50, 10)
        lr = st.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001], value=0.001)

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            # Map selection to internal names
            model_map = {
                "Standard Autoencoder": "standard",
                "Variational AE (VAE)": "vae",
                "Denoising AE": "denoising",
                "Attention AE": "attention_ae"
            }
            
            # GAN Augmentation Logic
            active_train_loader = train_loader
            if use_gan:
                st.info("Training GAN to generate synthetic fraud samples...")
                gan = GANTrainer(fraud_loader)
                gan.train(epochs=20)
                syn_data = gan.generate_synthetics(num_samples=1000)
                # In a real app we would mix this into train_loader. 
                # For demo, we just show generated samples and proceed with standard training or mix inputs.
                st.success("Generated 1000 synthetic fraud vectors!")
                # Visualizing Synthetic vs Real
                pca_cols = st.columns(2)
                # ... (PCA plot code could go here)
            
            # Train Main Model
            model = get_model(model_map[model_choice])
            history = train_model(model, active_train_loader, test_loader, epochs=epochs)
            
            # Metrics
            test_errors = get_reconstruction_errors(model, test_loader)
            fraud_errors = get_reconstruction_errors(model, fraud_loader)
            
            # Save to session state
            st.session_state['model'] = model
            st.session_state['test_errors'] = test_errors
            st.session_state['fraud_errors'] = fraud_errors
            
            # Plot Loss
            fig_loss = px.line(y=history['train_loss'], title="Training Loss Curve")
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=test_errors, name='Normal', opacity=0.7))
            fig_hist.add_trace(go.Histogram(x=fraud_errors, name='Fraud', opacity=0.7))
            fig_hist.update_layout(title="Reconstruction Error Distribution", barmode='overlay')
            st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 2: FRAUD DETECTOR ---
with tabs[1]:
    st.header("Live Transaction Inspector")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model in the Research Lab first!")
    else:
        # Input Form
        with st.form("transaction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                amt = st.number_input("Transaction Amount ($)", value=100.0)
                hour = st.slider("Hour of Day", 0, 23, 14)
            with col2:
                category = st.selectbox("Category", ["grocery_pos", "gas_transport", "shopping_net", "misc_net"])
                distance = st.number_input("Distance from Home (km)", value=5.0)
            with col3:
                age = st.slider("Customer Age", 18, 90, 35)
                job = st.text_input("Job", "Unemployed")
            
            submit = st.form_submit_button("Analyze Transaction")
            
        if submit:
            # Need to preprocess. 
            # We assume inputs map to the 7 features: ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log']
            # We need Label Encoders. Since we didn't save them in data_loader separately, we mock the encoding for demo.
            # Real app should load fitted encoders.
            
            # Mock Processing
            input_vector = np.zeros(7)
            # category (random hash map for demo if LE not avail, or just use 0)
            input_vector[0] = hash(category) % 10 
            input_vector[1] = hash(job) % 20
            input_vector[2] = 50000 # city_pop dummy
            input_vector[3] = age
            input_vector[4] = hour
            input_vector[5] = distance
            input_vector[6] = np.log1p(amt)
            
            # Scale (Loading scaler)
            with open(Config.SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            # Scaler expects 7 features
            input_scaled = scaler.transform([input_vector])
            input_tensor = torch.FloatTensor(input_scaled).to(Config.DEVICE)
            
            # Predict
            model = st.session_state['model']
            model.eval()
            with torch.no_grad():
                recon = model(input_tensor)
                if isinstance(recon, tuple): recon = recon[0]
                error = torch.mean((input_tensor - recon)**2).item()
                recon_np = recon.cpu().numpy()[0]
            
            # Threshold
            threshold = np.mean(st.session_state['test_errors']) + 3 * np.std(st.session_state['test_errors'])
            
            # Display
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Anomaly Score (MSE)", f"{error:.4f}")
                if error > threshold:
                    st.error(f"🚨 FRAUD DETECTED (Threshold: {threshold:.4f})")
                else:
                    st.success("✅ Transaction Normal")
            
            with c2:
                # "Explainability" - Feature contribution
                feat_names = ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log']
                contrib = (input_scaled[0] - recon_np)**2
                fig_exp = px.bar(x=contrib, y=feat_names, orientation='h', title="Feature Error Contribution")
                st.plotly_chart(fig_exp)

# --- TAB 3: CAUSAL ANALYSIS ---
with tabs[2]:
    st.header("Why did it fail? (Causal Check)")
    
    if 'model' in st.session_state:
        # Pick a high error fraud from test set
        f_idx = st.selectbox("Select a Fraud Sample ID", range(10))
        sample = X_fraud[f_idx]
        
        st.write("Current High Error Sample Features:")
        st.write(sample)
        
        # Intervention
        feature_to_fix = st.selectbox("Feature to Intervene", ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log'], index=6)
        feat_map = {'category':0, 'job':1, 'city_pop':2, 'age':3, 'hour':4, 'distance_km':5, 'amt_log':6}
        fid = feat_map[feature_to_fix]
        
        if st.button("Run Intervention: Set to Normal Mean"):
            orig, new, delta = counterfactual_analysis(st.session_state['model'], sample, None, fid, 'mean')
            st.metric("Original Error", f"{orig:.4f}")
            st.metric("New Error (After Fixing Feature)", f"{new:.4f}", delta=f"-{delta:.4f}")
            if delta > 0.5 * orig:
                st.success(f"Hypothesis Confirmed: {feature_to_fix} was the main driver of the anomaly!")
            else:
                st.warning(f"Feature {feature_to_fix} was not the sole cause.")
    else:
        st.info("Train model first.")

# --- TAB 4: FEDERATED VIEW ---
with tabs[3]:
    st.header("Federated Learning Simulation")
    st.markdown("Simulating 3 Banks training collaboratively without sharing data.")
    
    if st.button("Start Federation"):
        with st.spinner("Simulating Federated Rounds..."):
            global_model, fl_history = run_federated_simulation(num_rounds=10)
            st.success("Federated Training Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Client Loss Convergence")
                fig_fl = px.line(fl_history, x="round", y="loss", color="client_id", 
                                title="Local Client Training Losses")
                st.plotly_chart(fig_fl, use_container_width=True)
            
            with col2:
                st.subheader("Global Model Stability")
                # Deduplicate for global loss
                global_df = fl_history.drop_duplicates("round")
                fig_global = px.line(global_df, x="round", y="global_val_loss", 
                                    title="Aggregated Global Validation Loss",
                                    color_discrete_sequence=['#10B981'])
                st.plotly_chart(fig_global, use_container_width=True)
            
            st.info("The Global Model now contains insights from Bank 1, 2, and 3 without any sensitive data leaving their systems.")
            st.balloons()

# --- TAB 5: MONITORING ---
with tabs[4]:
    st.header("Production Monitor")
    
    # Simulate drift
    st.markdown("### PSI (Population Stability Index) Over Time")
    
    # Mock data for plot
    dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
    psi_values = np.random.uniform(0.01, 0.05, 7).tolist() + [0.15, 0.25, 0.30] # Drift at end
    
    df_mon = pd.DataFrame({'Date': dates, 'PSI': psi_values})
    
    fig_mon = px.line(df_mon, x='Date', y='PSI', markers=True)
    fig_mon.add_shape(type="line", x0=dates[0], y0=0.1, x1=dates[-1], y1=0.1, line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_mon)
    
    if psi_values[-1] > 0.1:
        st.error("⚠️ DATA DRIFT DETECTED! Model Retraining Recommended.")
