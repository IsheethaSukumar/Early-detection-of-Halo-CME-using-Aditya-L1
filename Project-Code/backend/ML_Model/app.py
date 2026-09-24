"""
Phase 2: FastAPI ML Inference & Data Microservice for Aditya-L1 Halo CME Forecasting
=====================================================================================
Usage:
    uvicorn Project-Code.ML_Model.app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Add ML_Model directory to Python module search path
ML_MODEL_DIR = Path(__file__).resolve().parent
if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

import json
import time
from typing import Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, status, Query
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from inference_pipeline import InferencePipeline
from db_manager import DatabaseManager
from alert_manager import AlertManager

# Load Metadata
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
META_PATH = ARTIFACT_DIR / "model_meta.json"

if not META_PATH.exists():
    raise FileNotFoundError("model_meta.json not found. Run export_artifacts.py first.")

with open(META_PATH, "r") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]
WARNING_THRESHOLD = META.get("warning_threshold", 0.37)
CRITICAL_THRESHOLD = META.get("critical_threshold", 0.85)

# Initialize Services
pipeline = InferencePipeline()
db = DatabaseManager()
alert_mgr = AlertManager()

app = FastAPI(
    title="Aditya-L1 Solar Weather & CME Forecasting API",
    description="Real-time microservice providing sub-second inference & monitoring for Halo CMEs.",
    version=META.get("model_version", "1.0.0"),
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class TelemetryRecord(BaseModel):
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")
    features: Dict[str, float] = Field(..., description="57 engineered features dictionary")


class PredictionResponse(BaseModel):
    timestamp: Optional[str]
    cme_probability: float
    halo_cme_predicted: bool
    risk_level: str
    is_anomaly: bool
    anomaly_score: float
    inference_latency: float
    base_probabilities: Dict[str, float]
    model_version: str


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "Aditya-L1 Halo CME Forecasting Microservice",
        "version": META["model_version"],
        "model_name": META["model_name"],
        "total_features": len(FEATURE_NAMES),
        "warning_threshold": WARNING_THRESHOLD,
        "critical_threshold": CRITICAL_THRESHOLD,
        "database": "PostgreSQL/TimescaleDB" if db.use_postgres else "SQLite",
    }


@app.get("/features", tags=["Metadata"])
def list_features():
    return {
        "total_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(payload: TelemetryRecord):
    t0 = time.time()
    try:
        ts = payload.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        prob, anomaly_score, is_anomaly, base_probs = pipeline.predict(payload.features)
        
        # Risk level determination
        if prob >= CRITICAL_THRESHOLD:
            risk_level = "CRITICAL"
        elif prob >= WARNING_THRESHOLD:
            risk_level = "WARNING"
        else:
            risk_level = "NORMAL"

        latency_ms = round((time.time() - t0) * 1000, 2)
        is_halo = bool(prob >= WARNING_THRESHOLD)

        # Trigger Alerts if Warning or Critical
        pred_dict = {
            "timestamp": ts,
            "cme_probability": prob,
            "risk_level": risk_level,
            "is_anomaly": is_anomaly
        }
        alert_result = alert_mgr.process_prediction(pred_dict)

        # Save to DB
        db.save_record(
            timestamp=ts,
            telemetry_dict=payload.features,
            cme_probability=prob,
            risk_level=risk_level,
            model_version=META["model_version"],
            inference_latency=latency_ms,
            anomaly_score=anomaly_score,
            anomaly_flag=is_anomaly,
            alert_info=alert_result
        )

        return PredictionResponse(
            timestamp=ts,
            cme_probability=prob,
            halo_cme_predicted=is_halo,
            risk_level=risk_level,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            inference_latency=latency_ms,
            base_probabilities=base_probs,
            model_version=META["model_version"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.get("/predict/sample", response_model=PredictionResponse, tags=["Inference Testing"])
def test_sample():
    sample_features = {col: 0.0 for col in FEATURE_NAMES}
    return predict_single(TelemetryRecord(timestamp="2026-09-05T18:00:00Z", features=sample_features))


@app.get("/latest", tags=["Monitoring"])
def get_latest_prediction():
    record = db.get_latest_prediction()
    if not record:
        raise HTTPException(status_code=404, detail="No predictions found in database.")
    return record


@app.get("/telemetry/latest", tags=["Monitoring"])
def get_latest_telemetry():
    record = db.get_latest_telemetry()
    if not record:
        raise HTTPException(status_code=404, detail="No telemetry records found.")
    return record


@app.get("/predictions", tags=["Monitoring"])
def get_predictions_history(limit: int = Query(default=50, ge=1, le=500)):
    records = db.get_historical_predictions(limit=limit)
    return {
        "count": len(records),
        "predictions": records
    }


@app.get("/status", tags=["System Status"])
def get_system_status():
    latest_pred = db.get_latest_prediction()
    latest_tel = db.get_latest_telemetry()
    return {
        "status": "OPERATIONAL",
        "model_version": META["model_version"],
        "database_backend": "PostgreSQL" if db.use_postgres else "SQLite",
        "last_telemetry_timestamp": latest_tel.get("timestamp") if latest_tel else None,
        "last_prediction_timestamp": latest_pred.get("timestamp") if latest_pred else None,
        "last_cme_probability": latest_pred.get("cme_probability") if latest_pred else None,
        "last_risk_level": latest_pred.get("risk_level") if latest_pred else None,
    }
