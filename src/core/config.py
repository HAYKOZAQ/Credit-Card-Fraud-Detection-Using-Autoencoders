import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
    VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    IMAGES_DIR = os.path.join(BASE_DIR, "visualizations") # Unified folder
    
    # Preprocessing
    CATEGORICAL_COLS = ['category', 'job']
    FEATURES = ['category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log']
    COLS_TO_DROP = ['trans_num', 'trans_date_trans_time', 'dob', 'merchant',
                    'city', 'state', 'lat', 'long', 'merch_lat', 'merch_long', 'amt']
    
    # Model Hyperparameters
    INPUT_DIM = len(FEATURES)
    HIDDEN_DIM1 = 5
    LATENT_DIM = 3
    
    # VAE Specific
    BETA = 0.01  # KL Divergence weight
    
    # Denoising AE Specific
    NOISE_FACTOR = 0.2
    
    # LSTM-AE Specific
    SEQ_LEN = 5  # Number of past transactions to consider
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 3
    
    # Financial Cost Model (Arbitrary weights)
    COST_FP = 10    # Cost of annoying a customer (False Positive)
    COST_FN = 100   # Cost of a missed fraud (False Negative)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42

# Create directories
for d in [Config.MODEL_DIR, Config.VIZ_DIR, Config.RESULTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
