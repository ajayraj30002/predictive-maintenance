# app.py - Complete Backend API
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import List, Dict

app = FastAPI(title="CMAPSS Predictive Maintenance API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
model = None
scaler = None
feature_names = None
max_rul = None

# Load artifacts on startup
@app.on_event("startup")
async def load_artifacts():
    global model, scaler, feature_names, max_rul
    
    # Load model and scaler
    model = joblib.load('artifacts/rul_model.joblib')
    scaler = joblib.load('artifacts/feature_scaler.joblib')
    
    # Load config
    with open('artifacts/model_config.json', 'r') as f:
        config = json.load(f)
        feature_names = config['feature_names']
        max_rul = config.get('max_rul', 125)
    
    print(f"✅ Model loaded. Expects {len(feature_names)} features")
    print(f"✅ Features: {feature_names[:5]}...")

# Request/Response models
class SensorData(BaseModel):
    cycle: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

class PredictionResponse(BaseModel):
    rul: float
    status: str
    confidence: str

@app.get("/")
async def root():
    return {"message": "CMAPSS Predictive Maintenance API", "status": "running"}

@app.get("/features")
async def get_features():
    """Return the list of features the model expects"""
    return {"feature_names": feature_names, "num_features": len(feature_names)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    """Predict RUL from sensor data"""
    try:
        # Convert to feature array in the exact order
        feature_dict = data.dict()
        feature_array = np.array([[feature_dict[feat] for feat in feature_names]])
        
        # Scale features
        features_scaled = scaler.transform(feature_array)
        
        # Predict
        rul = model.predict(features_scaled)[0]
        rul = max(0, min(rul, max_rul))  # Clip to [0, max_rul]
        
        # Determine status and confidence
        if rul > 100:
            status = "Healthy"
            confidence = "High"
        elif rul > 50:
            status = "Monitor"
            confidence = "Medium"
        elif rul > 20:
            status = "Warning"
            confidence = "Low"
        else:
            status = "Critical"
            confidence = "Very Low"
        
        return PredictionResponse(
            rul=round(rul, 2),
            status=status,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Upload CSV file with multiple samples and get predictions"""
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Check required columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns: {missing_cols[:5]}"
            )
        
        # Predict for all rows
        X = df[feature_names].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        predictions = np.clip(predictions, 0, max_rul)
        
        # Return results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "sample_id": i,
                "rul": round(pred, 2)
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend
@app.get("/frontend")
async def serve_frontend():
    return FileResponse("frontend/index.html")