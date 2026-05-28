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

tabs = st.tabs(["🏗️ Research Lab", "🕵️ Fraud Detector", "🔬 Causal Analysis", "🛡️ Robustness Suite", "🎯 Active Learning", "🕸️ Federated View", "📉 Monitoring"])

# --- TAB 1: RESEARCH LAB ---
with tabs[0]:
    st.header("Model Training & Experimentation")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Architecture", 
                                  ["Standard Autoencoder", "Variational AE (VAE)", "Denoising AE", "Attention AE", "Contrastive AE", "Graph AE"])
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
                "Attention AE": "attention_ae",
                "Contrastive AE": "contrastive",
                "Graph AE": "graph"
            }
            
            # GAN Augmentation Logic
            active_train_loader = train_loader
            if use_gan:
                st.info("Training GAN to generate synthetic fraud samples...")
                gan = GANTrainer(fraud_loader)
                gan.train(epochs=20)
                synthetic_data = gan.generate_synthetics(num_samples=len(fraud_loader.dataset))
                original_data = train_loader.dataset.data.numpy()
                augmented_data = np.concatenate([original_data, synthetic_data])
                from torch.utils.data import DataLoader, TensorDataset
                augmented_dataset = TensorDataset(torch.FloatTensor(augmented_data), torch.FloatTensor(augmented_data))
                active_train_loader = DataLoader(augmented_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
                st.success(f"Generated {len(synthetic_data)} synthetic fraud vectors!")
            
            # Train Main Model
            model = get_model(model_map[model_choice])
            history = train_model(model, active_train_loader, test_loader, epochs=epochs, lr=lr)
            model_name = model.__class__.__name__.lower()
            model_save_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pth")
            if os.path.exists(model_save_path):
                model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False))
            
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
        if not os.path.exists(Config.ENCODER_PATH):
            st.error("Encoder file not found. Run 'python main.py --mode research' first.")
        elif not os.path.exists(Config.SCALER_PATH):
            st.error("Scaler file not found. Run 'python main.py --mode research' first.")
        else:
            with open(Config.ENCODER_PATH, 'rb') as f:
                encoders = pickle.load(f)
            with open(Config.SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

            cat_options = list(encoders.get('category', {'classes_': ['0']}).classes_ if hasattr(encoders.get('category', object()), 'classes_') else ['0'])
            job_options = list(encoders.get('job', {'classes_': ['0']}).classes_ if hasattr(encoders.get('job', object()), 'classes_') else ['0'])

            with st.form("transaction_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    amt = st.number_input("Transaction Amount ($)", value=100.0)
                    hour = st.slider("Hour of Day", 0, 23, 14)
                with col2:
                    category = st.selectbox("Category", cat_options)
                    distance = st.number_input("Distance from Home (km)", value=5.0)
                with col3:
                    age = st.slider("Customer Age", 18, 90, 35)
                    city_pop = st.number_input("City Population", value=50000)
                    job = st.selectbox("Job", job_options)

                submit = st.form_submit_button("Analyze Transaction")

            if submit:
                stats = {}
                stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
                if os.path.exists(stats_path):
                    with open(stats_path, 'rb') as f:
                        stats = pickle.load(f)
                
                from src.core.data_loader import engineer_single_transaction
                df_single = engineer_single_transaction(
                    category=category,
                    job=job,
                    city_pop=city_pop,
                    age=age,
                    hour=hour,
                    distance_km=distance,
                    amt=amt,
                    stats=stats
                )
                
                for col in Config.CATEGORICAL_COLS:
                    le = encoders.get(col)
                    if le is not None:
                        val = str(df_single[col].iloc[0])
                        df_single[col] = le.transform([val])[0] if val in le.classes_ else le.transform([le.classes_[0]])[0]
                
                input_vector = df_single[Config.FEATURES].values
                input_scaled = scaler.transform(input_vector)
                input_tensor = torch.FloatTensor(input_scaled).to(Config.DEVICE)

                model = st.session_state['model']
                model.eval()
                with torch.no_grad():
                    if model.__class__.__name__ == 'GraphAutoencoder':
                        adj = torch.eye(1).to(Config.DEVICE)
                        recon = model(input_tensor, adj)
                    else:
                        recon = model(input_tensor)
                        
                    if isinstance(recon, tuple): 
                        recon = recon[0]
                    error = torch.mean((input_tensor - recon)**2).item()
                    recon_np = recon.cpu().numpy()[0]

                threshold = np.mean(st.session_state['test_errors']) + 3 * np.std(st.session_state['test_errors'])

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Anomaly Score (MSE)", f"{error:.4f}")
                    if error > threshold:
                        st.error(f"🚨 FRAUD DETECTED (Threshold: {threshold:.4f})")
                        
                        bg_path = os.path.join(Config.MODEL_DIR, "background_sample.pkl")
                        if os.path.exists(bg_path):
                            with open(bg_path, 'rb') as f:
                                background_data = pickle.load(f)
                        else:
                            background_data = np.zeros((50, Config.INPUT_DIM))
                        
                        from src.evaluation.explain import FraudExplainer
                        try:
                            explainer = FraudExplainer(model, background_data)
                            plot_path, _ = explainer.explain_instance(input_scaled[0], Config.FEATURES)
                            st.image(plot_path, use_container_width=True)
                        except Exception as e:
                            st.warning(f"SHAP explanation failed: {e}")
                    else:
                        st.success("✅ Transaction Normal")

                with c2:
                    feat_names = Config.FEATURES
                    contrib = (input_scaled[0] - recon_np)**2
                    fig_exp = px.bar(x=contrib, y=feat_names, orientation='h', title="Feature Error Contribution")
                    st.plotly_chart(fig_exp)

# --- TAB 3: CAUSAL ANALYSIS ---
with tabs[2]:
    st.header("Why did it fail? (Causal Check)")
    
    if 'model' in st.session_state:
        # Pick a high error fraud from test set
        max_fraud_idx = min(len(X_fraud) - 1, 99)
        f_idx = st.selectbox("Select a Fraud Sample ID", range(max_fraud_idx + 1))
        sample = X_fraud[f_idx]
        
        st.write("Current High Error Sample Features:")
        st.write(sample)
        
        # Intervention
        feature_to_fix = st.selectbox("Feature to Intervene", ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log'], index=6)
        feat_map = {'category':0, 'job':1, 'city_pop':2, 'age':3, 'hour':4, 'distance_km':5, 'amt_log':6}
        fid = feat_map[feature_to_fix]
        
        if st.button("Run Intervention: Set to Normal Mean"):
            orig, new, delta = counterfactual_analysis(st.session_state['model'], sample, target_feature_idx=fid, target_value='mean')
            st.metric("Original Error", f"{orig:.4f}")
            st.metric("New Error (After Fixing Feature)", f"{new:.4f}", delta=f"-{delta:.4f}")
            if delta > 0.5 * orig:
                st.success(f"Hypothesis Confirmed: {feature_to_fix} was the main driver of the anomaly!")
            else:
                st.warning(f"Feature {feature_to_fix} was not the sole cause.")
    else:
        st.info("Train model first.")

# --- TAB 4: ROBUSTNESS SUITE ---
with tabs[3]:
    st.header("🛡️ Adversarial Robustness Lab")
    st.markdown("Test model sensitivity against adversarial perturbations (FGSM Evasion attacks).")
    
    if 'model' in st.session_state:
        epsilon = st.slider("Perturbation Size (Epsilon)", 0.0, 0.5, 0.1, step=0.01)
        if st.button("Evaluate Adversarial Resilience"):
            with st.spinner("Generating adversarial examples..."):
                from src.evaluation.robustness import evaluate_robustness
                rob_results = evaluate_robustness({
                    'Active Model': st.session_state['model']
                }, test_loader, epsilon=epsilon)
                
                metrics = rob_results['Active Model']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clean Mean Error", f"{metrics['Clean Mean Error']:.4f}")
                    st.metric("Adversarial Mean Error", f"{metrics['Adversarial Mean Error (Max)']:.4f}")
                with col2:
                    st.metric("Evasion Mean Error", f"{metrics['Evasion Mean Error (Min)']:.4f}")
                    st.metric("Evasion Resistance", f"{metrics['Evasion Resistance (Min/Clean)']:.4f}")
                
                # Plot comparison
                fig_rob = go.Figure()
                fig_rob.add_trace(go.Bar(
                    x=['Clean Error', 'Adversarial Error', 'Evasion Error'],
                    y=[metrics['Clean Mean Error'], metrics['Adversarial Mean Error (Max)'], metrics['Evasion Mean Error (Min)']],
                    marker_color=['#3B82F6', '#EF4444', '#10B981']
                ))
                fig_rob.update_layout(title="Reconstruction Error Comparison Under Attack")
                st.plotly_chart(fig_rob)
    else:
        st.info("Train model first.")

# --- TAB 5: ACTIVE LEARNING ---
with tabs[4]:
    st.header("🎯 Active Learning Simulation")
    st.markdown("Simulate human-in-the-loop retraining to query highly anomalous transactions first.")
    
    if 'model' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            al_strategy = st.selectbox("AL Query Strategy", ["reconstruction_error", "random"])
        with col2:
            al_budget = st.slider("Labeling Budget", 50, 300, 100, step=10)
            
        if st.button("Run AL Simulation"):
            with st.spinner("Running Active Learning loop..."):
                from src.training.active_learning import ActiveLearningLab
                
                X_test_scaled = test_loader.dataset.data.numpy()
                X_fraud_scaled = fraud_loader.dataset.data.numpy()
                
                pool_X = np.concatenate([X_test_scaled, X_fraud_scaled])
                pool_y = np.concatenate([np.zeros(len(X_test_scaled)), np.ones(len(X_fraud_scaled))])
                
                # Shuffle
                idx = np.arange(len(pool_X))
                np.random.shuffle(idx)
                pool_X = pool_X[idx]
                pool_y = pool_y[idx]
                
                split = int(len(pool_X) * 0.5)
                al_X = pool_X[:split]
                al_y = pool_y[:split]
                eval_X = pool_X[split:]
                eval_y = pool_y[split:]
                
                lab = ActiveLearningLab(st.session_state['model'], al_X, al_y, eval_X, eval_y)
                al_results = lab.run_simulation(strategy=al_strategy, budget=al_budget, step_size=10)
                
                # Plot AUPRC learning curve
                fig_al = px.line(al_results, x='step', y='auprc', markers=True,
                                 title=f"Active Learning Curve ({al_strategy.upper()})",
                                 labels={'step': 'Labeled Samples Pool', 'auprc': 'Test set AUPRC'},
                                 color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig_al, use_container_width=True)
                st.success("AL Simulation Complete!")
    else:
        st.info("Train model first.")

# --- TAB 6: FEDERATED VIEW ---
with tabs[5]:
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

# --- TAB 7: MONITORING ---
with tabs[6]:
    st.header("Production Monitor")
    
    # Simulate drift
    st.markdown("### PSI (Population Stability Index) Over Time")
    
    if 'model' in st.session_state:
        ref_errors = st.session_state.get('test_errors', np.random.randn(500) * 0.1 + 0.5)
        prod_errors = st.session_state.get('fraud_errors', np.random.randn(500) * 0.2 + 0.8)
        simulated_psi = []
        dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
        for i in range(1, len(dates) + 1):
            drift_factor = 1.0 + (i / len(dates)) * 0.5
            drifted = ref_errors * drift_factor + np.random.randn(len(ref_errors)) * 0.05 * i
            simulated_psi.append(calculate_psi(ref_errors, drifted))
    else:
        dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
        simulated_psi = np.random.uniform(0.01, 0.05, 7).tolist() + [0.15, 0.25, 0.30]
    
    df_mon = pd.DataFrame({'Date': dates, 'PSI': simulated_psi})
    
    fig_mon = px.line(df_mon, x='Date', y='PSI', markers=True)
    fig_mon.add_shape(type="line", x0=dates[0], y0=0.1, x1=dates[-1], y1=0.1, line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_mon)
    
    if simulated_psi[-1] > 0.1:
        st.error("⚠️ DATA DRIFT DETECTED! Model Retraining Recommended.")
