"""
Production API for Credit Card Fraud Detection
FastAPI-based REST API for real-time fraud prediction
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from src.core.config import Config
from src.core.model import get_model
from src.core.data_loader import _engineer_features, Preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
encoders = None
merchant_stats = None
category_stats = None
state_stats = None
threshold = None

# API Key for authentication (in production, use proper secret management)
API_KEY = os.getenv("FRAUD_API_KEY", "your-secret-api-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for authentication"""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessing artifacts on startup"""
    global model, scaler, encoders, merchant_stats, category_stats, state_stats, threshold
    
    logger.info("Loading model and preprocessing artifacts...")
    
    try:
        # Load model
        model_path = os.path.join(Config.MODEL_DIR, "graphautoencoder.pth")
        if not os.path.exists(model_path):
            logger.warning(f"Graph AE not found, falling back to standard AE")
            model_path = os.path.join(Config.MODEL_DIR, "autoencoder.pth")
        
        model = get_model('graph' if 'graph' in model_path else 'standard')
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded")
        
        # Load encoders
        with open(Config.ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        logger.info("Encoders loaded")
        
        # Load statistics (if available)
        stats_path = os.path.join(Config.MODEL_DIR, "stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                merchant_stats = stats.get('merchant_stats')
                category_stats = stats.get('category_stats')
                state_stats = stats.get('state_stats')
            logger.info("Feature statistics loaded")
        else:
            logger.warning("Feature statistics not found, using defaults")
        
        # Calculate threshold from training data
        threshold = calculate_threshold()
        logger.info(f"Threshold calculated: {threshold:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise
    
    yield
    
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection using Graph Autoencoder with 25 features",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TransactionRequest(BaseModel):
    """Single transaction request"""
    trans_date_trans_time: str = Field(..., description="Transaction timestamp (YYYY-MM-DD HH:MM:SS)")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0, description="Transaction amount")
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State code")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    city_pop: int = Field(..., gt=0, description="City population")
    job: str = Field(..., description="Customer job")
    dob: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trans_date_trans_time": "2019-01-15 14:30:00",
                "merchant": "Test Merchant",
                "category": "shopping_net",
                "amt": 500.00,
                "city": "New York",
                "state": "NY",
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8336817,
                "job": "Software Engineer",
                "dob": "1985-06-15",
                "merch_lat": 40.7589,
                "merch_long": -73.9851
            }
        }


class BatchTransactionRequest(BaseModel):
    """Batch transaction request"""
    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):
    """Single prediction response"""
    is_fraud: bool = Field(..., description="Predicted fraud flag")
    fraud_probability: float = Field(..., description="Fraud probability score")
    reconstruction_error: float = Field(..., description="Reconstruction error from autoencoder")
    threshold: float = Field(..., description="Decision threshold")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    threshold: float
    timestamp: str


def calculate_threshold() -> float:
    """Calculate optimal threshold from training data"""
    try:
        # Load a sample of training data to calculate threshold
        # In production, this should be pre-calculated and stored
        import pandas as pd
        df = pd.read_csv(Config.DATA_PATH)
        df_normal = df[df['is_fraud'] == 0].sample(n=min(10000, len(df[df['is_fraud'] == 0])), random_state=42)
        
        # Engineer features
        df_normal = _engineer_features(
            df_normal.copy(),
            merchant_stats=merchant_stats,
            category_stats=category_stats,
            state_stats=state_stats
        )
        
        # Encode categorical features
        for col in Config.CATEGORICAL_COLS:
            if col in df_normal.columns and col in encoders:
                le = encoders[col]
                df_normal[col] = df_normal[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale features
        X_normal = df_normal[Config.FEATURES].values
        X_normal_scaled = scaler.transform(X_normal)
        
        # Calculate reconstruction errors
        X_tensor = torch.FloatTensor(X_normal_scaled).to(Config.DEVICE)
        with torch.no_grad():
            if hasattr(model, 'forward') and 'graph' in model.__class__.__name__.lower():
                # Graph AE needs edge_index
                n_nodes = X_tensor.shape[0]
                # Dummy self-loops for batch inference if true global graph isn't loaded
                edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).to(Config.DEVICE)
                recon, _ = model(X_tensor, edge_index)
            else:
                recon = model(X_tensor)
            
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        
        # Use 99th percentile as threshold (1% false positive rate)
        threshold = np.percentile(errors, 99)
        return float(threshold)
        
    except Exception as e:
        logger.error(f"Failed to calculate threshold: {e}")
        return 1.0  # Default fallback


def predict_single(transaction: TransactionRequest) -> PredictionResponse:
    """Predict fraud for a single transaction"""
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Engineer features
        df = _engineer_features(
            df,
            merchant_stats=merchant_stats,
            category_stats=category_stats,
            state_stats=state_stats
        )
        
        # Encode categorical features
        for col in Config.CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale features
        X = df[Config.FEATURES].values
        X_scaled = scaler.transform(X)
        
        # Predict
        X_tensor = torch.FloatTensor(X_scaled).to(Config.DEVICE)
        with torch.no_grad():
            if hasattr(model, 'forward') and 'graph' in model.__class__.__name__.lower():
                n_nodes = X_tensor.shape[0]
                edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).to(Config.DEVICE)
                recon, _ = model(X_tensor, edge_index)
            else:
                recon = model(X_tensor)
            
            error = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()[0]
        
        # Convert error to probability (sigmoid-like transformation)
        # Higher error = higher probability
        fraud_prob = 1 / (1 + np.exp(-(error - threshold) / (threshold * 0.1)))
        
        return PredictionResponse(
            is_fraud=bool(error > threshold),
            fraud_probability=float(fraud_prob),
            reconstruction_error=float(error),
            threshold=float(threshold),
            timestamp=datetime.now().isoformat(),
            model_version="2.0.0-graph-25features"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type=model.__class__.__name__ if model else "none",
        threshold=float(threshold) if threshold else 0.0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
async def predict(transaction: TransactionRequest):
    """
    Predict fraud for a single transaction
    
    Requires API key in X-API-Key header
    """
    return predict_single(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
async def predict_batch(request: BatchTransactionRequest):
    """
    Predict fraud for multiple transactions
    
    Requires API key in X-API-Key header
    """
    predictions = []
    for transaction in request.transactions:
        pred = predict_single(transaction)
        predictions.append(pred)
    
    fraud_count = sum(1 for p in predictions if p.is_fraud)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        fraud_count=fraud_count,
        fraud_rate=fraud_count / len(predictions) if predictions else 0.0
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    return {
        "model_type": model.__class__.__name__ if model else "none",
        "model_version": "2.0.0-graph-25features",
        "features_count": len(Config.FEATURES),
        "features": Config.FEATURES,
        "threshold": float(threshold) if threshold else 0.0,
        "device": str(Config.DEVICE)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
