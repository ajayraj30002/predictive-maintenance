# app.py - MLflow Version (Loads model from MLflow Server)
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
from typing import List, Dict

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# MLFLOW IMPORTS
# ============================================
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = FastAPI(title="CMAPSS Predictive Maintenance API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GLOBAL VARIABLES
# ============================================
model = None
scaler = None
feature_names = None
max_rul = 125

# ============================================
# MLFLOW CONFIGURATION (from environment)
# ============================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# ============================================
# CONFIG ENDPOINT FOR FRONTEND
# ============================================
@app.get("/config.js")
async def serve_config():
    """Serve config.js with environment variables injected"""
    api_url = os.getenv("API_URL", "https://predictive-maintenance-latest.onrender.com")
    content = f'window.ENV = {{ "API_URL": "{api_url}" }};'
    return Response(content, media_type="application/javascript")

# ============================================
# LOAD ARTIFACTS ON STARTUP
# ============================================
@app.on_event("startup")
async def load_artifacts():
    global model, scaler, feature_names, max_rul
    
    print("="*60)
    print("🚀 STARTING PREDICTIVE MAINTENANCE API (MLflow)")
    print("="*60)
    
    # Step 1: Configure MLflow
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set credentials for DagsHub
    if MLFLOW_USERNAME and MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
        print("✅ MLflow credentials configured")
    
    # Step 2: Load model from MLflow
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"📥 Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully from MLflow")
    except Exception as e:
        print(f"❌ Failed to load model from MLflow: {e}")
        print("   Make sure you've logged the model to MLflow first!")
        print("   Trying to load from local artifacts as fallback...")
        try:
            model = joblib.load('artifacts/rul_model.joblib')
            print("✅ Model loaded from local artifacts (fallback)")
        except:
            raise e
    
    # Step 3: Load scaler and config from local files
    try:
        scaler = joblib.load('artifacts/feature_scaler.joblib')
        print("✅ Scaler loaded from artifacts/")
    except Exception as e:
        print(f"⚠️ Scaler not found: {e}")
        scaler = None
    
    try:
        with open('artifacts/model_config.json', 'r') as f:
            config = json.load(f)
            feature_names = config.get('feature_names', [])
            max_rul = config.get('max_rul', 125)
        print(f"✅ Config loaded: {len(feature_names)} features")
    except Exception as e:
        print(f"⚠️ Config not found: {e}")
        feature_names = ['cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
                         'sensor_9', 'sensor_14', 'sensor_4', 'sensor_3', 
                         'sensor_17', 'sensor_7', 'sensor_12', 'sensor_2', 
                         'sensor_11', 'sensor_20']
    
    print("="*60)
    print("✅ API READY!")
    print("="*60)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
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

# ============================================
# ROUTES
# ============================================

@app.get("/")
async def root():
    return {
        "message": "CMAPSS Predictive Maintenance API",
        "status": "running",
        "model_source": "MLflow" if model else "Not loaded"
    }

@app.get("/features")
async def get_features():
    """Return the list of features the model expects"""
    return {"feature_names": feature_names, "num_features": len(feature_names)}

@app.get("/model-info")
async def get_model_info():
    """Return model information from MLflow"""
    try:
        client = MlflowClient()
        model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if model_version:
            v = model_version[0]
            return {
                "model_name": MODEL_NAME,
                "stage": MODEL_STAGE,
                "version": v.version,
                "run_id": v.run_id,
                "status": v.status
            }
        return {"message": "No model version found"}
    except:
        return {"message": "MLflow info not available"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    """Predict RUL from sensor data"""
    try:
        # Convert to feature array in the exact order
        feature_dict = data.dict()
        feature_array = np.array([[feature_dict[feat] for feat in feature_names]])
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(feature_array)
        else:
            features_scaled = feature_array
        
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
        X_scaled = scaler.transform(X) if scaler else X
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

# Health check with model info
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": "MLflow" if model else "None",
        "features": len(feature_names) if feature_names else 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
