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

def _engineer_features(df, merchant_stats=None, category_stats=None, state_stats=None):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['distance_km'] = haversine_distance(df['lat'], df['long'],
                                           df['merch_lat'], df['merch_long'])
    df['amt_log'] = np.log1p(df['amt'])
    
    # Temporal features
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['trans_date_trans_time'].dt.day <= 5).astype(int)
    df['is_month_end'] = (df['trans_date_trans_time'].dt.day >= 25).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Geographic features
    df['city_pop_log'] = np.log1p(df['city_pop'])
    df['is_rural'] = (df['city_pop'] < 10000).astype(int)
    
    # Amount-based features (require statistics)
    if category_stats is not None:
        df['category_mean'] = df['category'].map(category_stats.get('mean', {}))
        df['category_std'] = df['category'].map(category_stats.get('std', {}))
        df['category_median'] = df['category'].map(category_stats.get('median', {}))
        df['category_95th'] = df['category'].map(category_stats.get('95th', {}))
        
        df['amt_zscore_category'] = (df['amt'] - df['category_mean']) / (df['category_std'] + 1e-6)
        df['amt_ratio_to_median'] = df['amt'] / (df['category_median'] + 1e-6)
        df['is_high_amount'] = (df['amt'] > df['category_95th']).astype(int)
        
        df.drop(['category_mean', 'category_std', 'category_median', 'category_95th'], axis=1, inplace=True)
    else:
        df['amt_zscore_category'] = 0
        df['amt_ratio_to_median'] = 1
        df['is_high_amount'] = 0
    
    if merchant_stats is not None:
        df['merchant_mean'] = df['merchant'].map(merchant_stats.get('mean', {}))
        df['merchant_std'] = df['merchant'].map(merchant_stats.get('std', {}))
        df['amt_zscore_merchant'] = (df['amt'] - df['merchant_mean']) / (df['merchant_std'] + 1e-6)
        df['merchant_fraud_rate'] = df['merchant'].map(merchant_stats.get('fraud_rate', {}))
        df['merchant_transaction_volume'] = df['merchant'].map(merchant_stats.get('volume', {}))
        df['merchant_avg_amount'] = df['merchant'].map(merchant_stats.get('mean', {}))
        
        df.drop(['merchant_mean', 'merchant_std'], axis=1, inplace=True)
    else:
        df['amt_zscore_merchant'] = 0
        df['merchant_fraud_rate'] = 0
        df['merchant_transaction_volume'] = 0
        df['merchant_avg_amount'] = 0
    
    if state_stats is not None:
        df['state_fraud_rate'] = df['state'].map(state_stats.get('fraud_rate', {}))
    else:
        df['state_fraud_rate'] = 0
    
    # Risk indicators
    high_risk_categories = ['shopping_net', 'grocery_pos', 'misc_net', 'shopping_pos']
    df['is_high_risk_category'] = df['category'].isin(high_risk_categories).astype(int)
    df['is_night_transaction'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    
    return df

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.merchant_stats = None
        self.category_stats = None
        self.state_stats = None

    def fit_transform(self, df):
        print("Processing dates and calculating features...")
        
        # Compute statistics from full dataset
        self._compute_statistics(df)
        
        df = _engineer_features(df, self.merchant_stats, self.category_stats, self.state_stats)

        drop_cols = [c for c in Config.COLS_TO_DROP if c in df.columns]
        df_clean = df.drop(columns=drop_cols)

        for col in Config.CATEGORICAL_COLS:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        with open(Config.ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        normal_data = df_clean[df_clean['is_fraud'] == 0]
        fraud_data = df_clean[df_clean['is_fraud'] == 1]

        train_df, test_df = train_test_split(normal_data, test_size=0.2, random_state=Config.SEED)
        
        X_train = train_df[Config.FEATURES].values
        X_test = test_df[Config.FEATURES].values
        X_fraud_vals = fraud_data[Config.FEATURES].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_fraud_scaled = self.scaler.transform(X_fraud_vals)
        
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'merchant_stats': self.merchant_stats,
                'category_stats': self.category_stats,
                'state_stats': self.state_stats
            }, f)
        
        return X_train_scaled, X_test_scaled, X_fraud_scaled, train_df, test_df, fraud_data
    
    def _compute_statistics(self, df):
        # Category statistics
        self.category_stats = {
            'mean': df.groupby('category')['amt'].mean().to_dict(),
            'std': df.groupby('category')['amt'].std().to_dict(),
            'median': df.groupby('category')['amt'].median().to_dict(),
            '95th': df.groupby('category')['amt'].quantile(0.95).to_dict()
        }
        
        # Merchant statistics
        self.merchant_stats = {
            'mean': df.groupby('merchant')['amt'].mean().to_dict(),
            'std': df.groupby('merchant')['amt'].std().to_dict(),
            'fraud_rate': df.groupby('merchant')['is_fraud'].mean().to_dict(),
            'volume': df.groupby('merchant').size().to_dict()
        }
        
        # State statistics
        self.state_stats = {
            'fraud_rate': df.groupby('state')['is_fraud'].mean().to_dict()
        }


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

def engineer_single_transaction(category, job, city_pop, age, hour, distance_km, amt, stats, merchant='unknown', state='unknown', trans_date=None):
    if trans_date is None:
        trans_date = pd.Timestamp.now()
    else:
        trans_date = pd.to_datetime(trans_date)
        
    df = pd.DataFrame([{
        'category': category,
        'job': job,
        'city_pop': city_pop,
        'age': age,
        'hour': hour,
        'distance_km': distance_km,
        'amt': amt,
        'merchant': merchant,
        'state': state,
        'trans_date_trans_time': trans_date
    }])
    
    # Calculate log amount
    df['amt_log'] = np.log1p(df['amt'])
    
    # Temporal features
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['trans_date_trans_time'].dt.day <= 5).astype(int)
    df['is_month_end'] = (df['trans_date_trans_time'].dt.day >= 25).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Geographic features
    df['city_pop_log'] = np.log1p(df['city_pop'])
    df['is_rural'] = (df['city_pop'] < 10000).astype(int)
    
    # Category stats
    cat_stats = stats.get('category_stats', {}) if stats else {}
    if cat_stats:
        cat_mean = cat_stats.get('mean', {}).get(category, 0)
        cat_std = cat_stats.get('std', {}).get(category, 1)
        cat_median = cat_stats.get('median', {}).get(category, 0)
        cat_95th = cat_stats.get('95th', {}).get(category, 0)
        
        df['amt_zscore_category'] = (df['amt'] - cat_mean) / (cat_std + 1e-6)
        df['amt_ratio_to_median'] = df['amt'] / (cat_median + 1e-6)
        df['is_high_amount'] = (df['amt'] > cat_95th).astype(int)
    else:
        df['amt_zscore_category'] = 0
        df['amt_ratio_to_median'] = 1
        df['is_high_amount'] = 0
        
    # Merchant stats
    merch_stats = stats.get('merchant_stats', {}) if stats else {}
    if merch_stats:
        merch_mean = merch_stats.get('mean', {}).get(merchant, 0)
        merch_std = merch_stats.get('std', {}).get(merchant, 1)
        df['amt_zscore_merchant'] = (df['amt'] - merch_mean) / (merch_std + 1e-6)
        df['merchant_fraud_rate'] = merch_stats.get('fraud_rate', {}).get(merchant, 0)
        df['merchant_transaction_volume'] = merch_stats.get('volume', {}).get(merchant, 0)
        df['merchant_avg_amount'] = merch_stats.get('mean', {}).get(merchant, 0)
    else:
        df['amt_zscore_merchant'] = 0
        df['merchant_fraud_rate'] = 0
        df['merchant_transaction_volume'] = 0
        df['merchant_avg_amount'] = 0
        
    # State stats
    state_stats = stats.get('state_stats', {}) if stats else {}
    if state_stats:
        df['state_fraud_rate'] = state_stats.get('fraud_rate', {}).get(state, 0)
    else:
        df['state_fraud_rate'] = 0
        
    # Risk indicators
    high_risk_categories = ['shopping_net', 'grocery_pos', 'misc_net', 'shopping_pos']
    df['is_high_risk_category'] = df['category'].isin(high_risk_categories).astype(int)
    df['is_night_transaction'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    
    return df

def create_sequences(df, scaler, stats=None):
    """
    Creates sequences for LSTM. Use synthetic 'cust_id' since cc_num is missing.
    """
    print("Creating transaction sequences for LSTM (using synthetic cust_id)...")
    df['cust_id'] = df['city_pop'].astype(str) + '_' + df['job'].astype(str) + '_' + df['lat'].astype(str) + '_' + df['long'].astype(str) + '_' + df['dob'].astype(str)
    
    if stats is None:
        stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                
    if stats is not None:
        df = _engineer_features(df,
                                merchant_stats=stats.get('merchant_stats'),
                                category_stats=stats.get('category_stats'),
                                state_stats=stats.get('state_stats'))
    else:
        df = _engineer_features(df)
        
    df = df.sort_values(['cust_id', 'trans_date_trans_time'])
    
    features = Config.FEATURES
    # Load encoders
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    for col in Config.CATEGORICAL_COLS:
        if col in encoders:
            raw_vals = df[col].astype(str)
            known = set(encoders[col].classes_)
            safe_vals = raw_vals.apply(lambda x: x if x in known else encoders[col].classes_[0])
            df[col] = encoders[col].transform(safe_vals)
    
    scaled_feats = scaler.transform(df[features].values)
    df_scaled = pd.DataFrame(scaled_feats, columns=features)
    df_scaled['cust_id'] = df['cust_id'].values
    
    sequences = []
    all_custs = df_scaled['cust_id'].unique()
    processed_custs = min(len(all_custs), 1000)
    if len(all_custs) > 1000:
        print(f"Note: Only processing {processed_custs}/{len(all_custs)} customers for sequence creation.")
    for cust in all_custs[:processed_custs]:
        group = df_scaled[df_scaled['cust_id'] == cust]
        if len(group) >= Config.SEQ_LEN:
            data = group[features].values
            for i in range(len(data) - Config.SEQ_LEN + 1):
                sequences.append(data[i:i+Config.SEQ_LEN])
    
    if not sequences:
        raise ValueError("No sequences created! Check that SEQ_LEN <= min transaction counts per customer.")
        
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
    if not os.path.exists(Config.SCALER_PATH) or not os.path.exists(Config.ENCODER_PATH):
        raise FileNotFoundError(
            "Scaler/encoder files not found. Run 'python main.py --mode research' first.")
    df = pd.read_csv(Config.DATA_PATH)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
    normal_df = df[df['is_fraud'] == 0]
    fraud_df = df[df['is_fraud'] == 1]
    
    # Take larger chunks for normal sequence training
    # Use dynamic splitting based on length
    n_normal = len(normal_df)
    train_end = int(n_normal * 0.8)
    
    train_seq = create_sequences(normal_df.iloc[:train_end].copy(), scaler, stats)
    test_seq = create_sequences(normal_df.iloc[train_end:].copy(), scaler, stats)
    fraud_seq = create_sequences(fraud_df.copy(), scaler, stats)
    
    return (DataLoader(SequenceDataset(train_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(test_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(fraud_seq), batch_size=Config.BATCH_SIZE))

def get_merchant_loaders():
    """
    Creates sequences grouped by merchant. 
    """
    if not os.path.exists(Config.SCALER_PATH) or not os.path.exists(Config.ENCODER_PATH):
        raise FileNotFoundError(
            "Scaler/encoder files not found. Run 'python main.py --mode research' first.")
    df = pd.read_csv(Config.DATA_PATH)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    
    print("Creating transaction sequences per Merchant...")
    df = _engineer_features(df, 
                            merchant_stats=stats.get('merchant_stats') if stats else None,
                            category_stats=stats.get('category_stats') if stats else None,
                            state_stats=stats.get('state_stats') if stats else None)
    
    features = Config.FEATURES
    # Load encoders
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    for col in Config.CATEGORICAL_COLS:
        if col in encoders:
            raw_vals = df[col].astype(str)
            known = set(encoders[col].classes_)
            safe_vals = raw_vals.apply(lambda x: x if x in known else encoders[col].classes_[0])
            df[col] = encoders[col].transform(safe_vals)
        
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
        raise ValueError("No merchant sequences created! Check that SEQ_LEN <= min transaction counts per merchant.")
        
    # Simple split for demonstration (80/20)
    full_seq = np.array(sequences)
    split_idx = int(len(full_seq) * 0.8)
    train_seq = full_seq[:split_idx]
    test_seq = full_seq[split_idx:]
    
    return (DataLoader(SequenceDataset(train_seq), batch_size=Config.BATCH_SIZE),
            DataLoader(SequenceDataset(test_seq), batch_size=Config.BATCH_SIZE))

def get_pyg_loaders():
    """
    Creates a global PyTorch Geometric graph and returns NeighborLoaders.
    Nodes are transactions, edges connect transactions by the same customer and merchant.
    """
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
    except ImportError:
        raise ImportError("Please install torch-geometric to use GNN models.")

    if not os.path.exists(Config.SCALER_PATH) or not os.path.exists(Config.ENCODER_PATH):
        raise FileNotFoundError("Scaler/encoder files not found. Run 'python main.py --mode research' first.")
    
    df = pd.read_csv(Config.DATA_PATH)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

    print("Creating Global Transaction Graph for GNN...")
    
    # Engineer features
    df['cust_id'] = df['city_pop'].astype(str) + '_' + df['job'].astype(str) + '_' + df['lat'].astype(str) + '_' + df['long'].astype(str) + '_' + df['dob'].astype(str)
    
    df = _engineer_features(df, 
                            merchant_stats=stats.get('merchant_stats') if stats else None,
                            category_stats=stats.get('category_stats') if stats else None,
                            state_stats=stats.get('state_stats') if stats else None)
    
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    df['node_id'] = df.index
    
    # Features & Scaling
    features = Config.FEATURES
    with open(Config.ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)

    for col in Config.CATEGORICAL_COLS:
        if col in encoders:
            raw_vals = df[col].astype(str)
            known = set(encoders[col].classes_)
            safe_vals = raw_vals.apply(lambda x: x if x in known else encoders[col].classes_[0])
            df[col] = encoders[col].transform(safe_vals)
            
    scaled_feats = scaler.transform(df[features].values)
    x = torch.FloatTensor(scaled_feats)
    y = torch.FloatTensor(df['is_fraud'].values)
    
    # Build Edges (Sparse)
    source_nodes = []
    target_nodes = []
    
    # 1. Customer edges (sequential)
    print("Building customer edges...")
    df['prev_node_cust'] = df.groupby('cust_id')['node_id'].shift(1)
    mask_cust = df['prev_node_cust'].notna()
    source_nodes.extend(df.loc[mask_cust, 'prev_node_cust'].astype(int).tolist())
    target_nodes.extend(df.loc[mask_cust, 'node_id'].astype(int).tolist())
    
    # 2. Merchant edges (sequential)
    print("Building merchant edges...")
    df['prev_node_merch'] = df.groupby('merchant')['node_id'].shift(1)
    mask_merch = df['prev_node_merch'].notna()
    source_nodes.extend(df.loc[mask_merch, 'prev_node_merch'].astype(int).tolist())
    target_nodes.extend(df.loc[mask_merch, 'node_id'].astype(int).tolist())
    
    # Make undirected
    src = source_nodes + target_nodes
    dst = target_nodes + source_nodes
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Split nodes into train/test/fraud based on original logic roughly
    # To keep it simple: normal nodes for training (80%), normal nodes for testing (20%), fraud nodes for testing
    is_fraud = df['is_fraud'] == 1
    normal_idx = df.index[~is_fraud].values
    fraud_idx = df.index[is_fraud].values
    
    np.random.seed(Config.SEED)
    np.random.shuffle(normal_idx)
    
    split_point = int(len(normal_idx) * 0.8)
    train_idx = normal_idx[:split_point]
    test_idx = normal_idx[split_point:]
    
    # Loaders
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], input_nodes=torch.tensor(train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = NeighborLoader(data, num_neighbors=[10, 10], input_nodes=torch.tensor(test_idx), batch_size=Config.BATCH_SIZE)
    fraud_loader = NeighborLoader(data, num_neighbors=[10, 10], input_nodes=torch.tensor(fraud_idx), batch_size=Config.BATCH_SIZE)
    
    return train_loader, test_loader, fraud_loader

