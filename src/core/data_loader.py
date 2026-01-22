import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from src.core.config import Config

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df):
        print("Processing dates and calculating features...")
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])

        # Feature 1: Age of customer
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

        # Feature 2: Hour of transaction
        df['hour'] = df['trans_date_trans_time'].dt.hour

        # Feature 3: Distance in KM
        df['distance_km'] = haversine_distance(df['lat'], df['long'],
                                               df['merch_lat'], df['merch_long'])

        # Feature 4: Log Scale the Amount
        df['amt_log'] = np.log1p(df['amt'])

        drop_cols = [c for c in Config.COLS_TO_DROP if c in df.columns]
        df_clean = df.drop(columns=drop_cols)

        # Encode categorical data
        for col in Config.CATEGORICAL_COLS:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        # Save encoders and scaler
        with open(Config.ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Split into normal and fraud
        normal_data = df_clean[df_clean['is_fraud'] == 0]
        fraud_data = df_clean[df_clean['is_fraud'] == 1]

        # Use Config.FEATURES for consistency
        train_df, test_df = train_test_split(normal_data, test_size=0.2, random_state=Config.SEED)
        
        X_train = train_df[Config.FEATURES].values
        X_test = test_df[Config.FEATURES].values
        X_fraud_vals = fraud_data[Config.FEATURES].values
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_fraud_scaled = self.scaler.transform(X_fraud_vals)
        
        # Save the scaler
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X_train_scaled, X_test_scaled, X_fraud_scaled, train_df, test_df, fraud_data


class FraudDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]

def create_sequences(df, scaler):
    """
    Creates sequences for LSTM. Use synthetic 'cust_id' since cc_num is missing.
    """
    print("Creating transaction sequences for LSTM (using synthetic cust_id)...")
    # Synthetic ID: city_pop + job + lat + long
    df['cust_id'] = df['city_pop'].astype(str) + df['job'] + df['lat'].astype(str)
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values(['cust_id', 'trans_date_trans_time'])
    
    # Feature engineering
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['amt_log'] = np.log1p(df['amt'])
    
    features = Config.FEATURES
    # Load encoders
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    for col in Config.CATEGORICAL_COLS:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    scaled_feats = scaler.transform(df[features].values)
    df_scaled = pd.DataFrame(scaled_feats, columns=features)
    df_scaled['cust_id'] = df['cust_id'].values
    
    sequences = []
    # Limit number of customers processed for speed
    all_custs = df_scaled['cust_id'].unique()
    for cust in all_custs[:1000]: # Sample 1000 customers
        group = df_scaled[df_scaled['cust_id'] == cust]
        if len(group) >= Config.SEQ_LEN:
            data = group[features].values
            for i in range(len(data) - Config.SEQ_LEN + 1):
                sequences.append(data[i:i+Config.SEQ_LEN])
    
    if not sequences:
        print("Warning: No sequences created!")
        return np.zeros((1, Config.SEQ_LEN, len(features)))
        
    return np.array(sequences)

def get_dataloaders():
    df = pd.read_csv(Config.DATA_PATH)
    preprocessor = Preprocessor()
    X_train, X_test, X_fraud, train_df, test_df, fraud_df = preprocessor.fit_transform(df.copy())

    train_dataset = FraudDataset(X_train)
    test_dataset = FraudDataset(X_test)
    fraud_dataset = FraudDataset(X_fraud)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    fraud_loader = DataLoader(fraud_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, fraud_loader, X_fraud, train_df, test_df, fraud_df


def get_sequential_loaders():
    df = pd.read_csv(Config.DATA_PATH)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    normal_df = df[df['is_fraud'] == 0]
    fraud_df = df[df['is_fraud'] == 1]
    
    # Take larger chunks for normal sequence training
    # Use dynamic splitting based on length
    n_normal = len(normal_df)
    train_end = int(n_normal * 0.8)
    
    train_seq = create_sequences(normal_df.iloc[:train_end].copy(), scaler)
    test_seq = create_sequences(normal_df.iloc[train_end:].copy(), scaler)
    fraud_seq = create_sequences(fraud_df.copy(), scaler)
    
    return (DataLoader(SequenceDataset(train_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(test_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(fraud_seq), batch_size=Config.BATCH_SIZE))

def get_merchant_loaders():
    """
    Creates sequences grouped by merchant. 
    Using 'city_pop' + 'merch_lat' as a proxy for merchant ID if 'merchant' col is dropped early or not unique enough.
    Actually 'merchant' column exists in raw data.
    """
    df = pd.read_csv(Config.DATA_PATH)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # We need to process features but group by merchant
    # Re-using create_sequences reasoning but grouping by merchant
    print("Creating transaction sequences per Merchant...")
    
    # Pre-process for features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['amt_log'] = np.log1p(df['amt'])
    
    features = Config.FEATURES
    # Load encoders
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    for col in Config.CATEGORICAL_COLS:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
        
    scaled_feats = scaler.transform(df[features].values)
    df_scaled = pd.DataFrame(scaled_feats, columns=features)
    # Add merchant info back
    df_scaled['merchant'] = df['merchant'].values
    
    sequences = []
    # Sample top 500 merchants by volume
    top_merchants = df['merchant'].value_counts().head(500).index
    
    for merch in top_merchants:
        group = df_scaled[df_scaled['merchant'] == merch]
        if len(group) >= Config.SEQ_LEN:
            data = group[features].values
            for i in range(len(data) - Config.SEQ_LEN + 1):
                sequences.append(data[i:i+Config.SEQ_LEN])
                
    if not sequences:
        print("Warning: No merchant sequences created!")
        return DataLoader(SequenceDataset(np.zeros((1, Config.SEQ_LEN, len(features)))), batch_size=Config.BATCH_SIZE)
        
    # Simple split for demonstration (80/20)
    full_seq = np.array(sequences)
    split_idx = int(len(full_seq) * 0.8)
    train_seq = full_seq[:split_idx]
    test_seq = full_seq[split_idx:]
    
    return (DataLoader(SequenceDataset(train_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(test_seq), batch_size=Config.BATCH_SIZE))

