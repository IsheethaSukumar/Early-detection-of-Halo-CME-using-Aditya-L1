"""
Phases 5 & 6: Unified Inference & Stacking Ensemble Pipeline
============================================================
Handles:
1. Feature scaling (StandardScaler)
2. Base Model predictions: Random Forest, XGBoost, BiLSTM (PyTorch)
3. Meta-Model Stacking (Logistic Regression) -> cme_probability
4. Isolation Forest parallel execution -> anomaly_score & is_anomaly
"""

import json
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
META_PATH = ARTIFACT_DIR / "model_meta.json"

with open(META_PATH, "r") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]

# PyTorch BiLSTM Architecture definition
if TORCH_AVAILABLE:
    class HaloCME_BiLSTM_IF(nn.Module):
        def __init__(self, n_features: int = 59, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.3):
            super().__init__()
            self.input_bn = nn.BatchNorm1d(n_features)
            self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True, bidirectional=True)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=True)
            self.drop2 = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden2 * 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, s, f = x.shape
            x = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            out = self.drop2(out)
            return self.fc(out[:, -1, :])


class InferencePipeline:
    def __init__(self):
        print("[InferencePipeline] Loading model artifacts...")
        # 1. Primary RF & Scaler
        rf_path = ARTIFACT_DIR / META.get("model_file", "random_forest_tuned.pkl")
        if not rf_path.exists():
            rf_path = ARTIFACT_DIR / "random_forest.pkl"
        self.rf_model = joblib.load(rf_path)
        
        scaler_path = ARTIFACT_DIR / META.get("scaler_file", "rf_scaler.pkl")
        self.scaler = joblib.load(scaler_path)

        # 2. XGBoost
        xgb_path = ARTIFACT_DIR / "xgboost_model.pkl"
        if not xgb_path.exists():
            xgb_path = ARTIFACT_DIR / "xgb_if_model.pkl"
        self.xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None

        # 3. BiLSTM
        self.bilstm_model = None
        bilstm_path = ARTIFACT_DIR / "lstm_if_model.pt"
        if not bilstm_path.exists():
            bilstm_path = ARTIFACT_DIR / "lstm_model.pt"
        if TORCH_AVAILABLE and bilstm_path.exists():
            try:
                device = torch.device("cpu")
                model = HaloCME_BiLSTM_IF(n_features=59)
                model.load_state_dict(torch.load(bilstm_path, map_location=device))
                model.eval()
                self.bilstm_model = model
                print("[InferencePipeline] Loaded PyTorch BiLSTM model.")
            except Exception as e:
                print(f"[InferencePipeline Warning] Could not load PyTorch model: {e}")

        # 4. Stacking Meta-model
        meta_path = ARTIFACT_DIR / "stack_meta_model.pkl"
        self.meta_model = joblib.load(meta_path) if meta_path.exists() else None
        
        meta_scaler_path = ARTIFACT_DIR / "stack_meta_scaler.pkl"
        self.meta_scaler = joblib.load(meta_scaler_path) if meta_scaler_path.exists() else None

        # 5. Isolation Forest
        iforest_path = ARTIFACT_DIR / META.get("iforest_file", "iforest_model.pkl")
        self.iforest_model = joblib.load(iforest_path) if iforest_path.exists() else None

    def preprocess(self, input_dict: Dict[str, float]) -> np.ndarray:
        record_vals = [float(input_dict.get(col, 0.0)) for col in FEATURE_NAMES]
        df_input = pd.DataFrame([record_vals], columns=FEATURE_NAMES)
        return self.scaler.transform(df_input)

    def predict(self, input_dict: Dict[str, float], window_records: Optional[list] = None) -> Tuple[float, float, bool, Dict[str, float]]:
        scaled_features = self.preprocess(input_dict)

        # 1. RF Base Prediction
        rf_prob = float(self.rf_model.predict_proba(scaled_features)[0][1])

        # 2. XGBoost Base Prediction
        xgb_prob = rf_prob
        if self.xgb_model is not None:
            try:
                xgb_prob = float(self.xgb_model.predict_proba(scaled_features)[0][1])
            except Exception:
                pass

        # 3. Isolation Forest Anomaly Detection
        anomaly_score = 0.0
        is_anomaly = False
        if self.iforest_model is not None:
            try:
                score = float(self.iforest_model.score_samples(scaled_features)[0])
                anomaly_score = round(score, 4)
                is_anomaly = bool(self.iforest_model.predict(scaled_features)[0] == -1)
            except Exception:
                pass

        # 4. BiLSTM Base Prediction
        bilstm_prob = (rf_prob + xgb_prob) / 2.0
        if self.bilstm_model is not None:
            try:
                # Construct sequence of 24 time steps (57 solar features + anomaly_score + is_anomaly)
                # If window_records available, use them; otherwise duplicate single record
                single_vec = np.hstack([scaled_features[0], [anomaly_score, float(is_anomaly)]])
                seq_vec = np.tile(single_vec, (1, 24, 1)).astype(np.float32)
                with torch.no_grad():
                    logit = self.bilstm_model(torch.from_numpy(seq_vec))
                    bilstm_prob = float(torch.sigmoid(logit).item())
            except Exception as e:
                pass

        # 5. Stacking Meta-model or Weighted Ensemble
        if self.meta_model is not None:
            try:
                oof_row = np.array([[rf_prob, xgb_prob, bilstm_prob]])
                if self.meta_scaler is not None:
                    oof_row = self.meta_scaler.transform(oof_row)
                final_prob = float(self.meta_model.predict_proba(oof_row)[0][1])
            except Exception:
                final_prob = 0.5 * rf_prob + 0.3 * xgb_prob + 0.2 * bilstm_prob
        else:
            final_prob = 0.5 * rf_prob + 0.3 * xgb_prob + 0.2 * bilstm_prob

        base_probs = {
            "rf_prob": round(rf_prob, 4),
            "xgb_prob": round(xgb_prob, 4),
            "bilstm_prob": round(bilstm_prob, 4)
        }

        return round(final_prob, 4), anomaly_score, is_anomaly, base_probs
