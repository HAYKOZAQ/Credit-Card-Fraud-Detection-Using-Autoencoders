import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
    VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Preprocessing
    CATEGORICAL_COLS = ['category', 'job']
    FEATURES = [
        'category', 'job', 'city_pop', 'age', 'hour', 'distance_km', 'amt_log',
        'amt_zscore_category', 'amt_zscore_merchant', 'amt_ratio_to_median',
        'is_high_amount', 'is_high_risk_category', 'is_night_transaction',
        'is_weekend', 'merchant_fraud_rate', 'merchant_transaction_volume',
        'merchant_avg_amount', 'day_of_week', 'is_month_start', 'is_month_end',
        'hour_sin', 'hour_cos', 'state_fraud_rate', 'city_pop_log', 'is_rural'
    ]
    COLS_TO_DROP = ['trans_num', 'trans_date_trans_time', 'dob', 'merchant',
                    'city', 'state', 'lat', 'long', 'merch_lat', 'merch_long', 'amt']
    
    # Model Hyperparameters
    INPUT_DIM = len(FEATURES)
    HIDDEN_DIM1 = 64
    HIDDEN_DIM2 = 32
    LATENT_DIM = 16
    
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
