"""
Phase 3 & 4: Real-Time Stream Worker & Feature Buffer Engine
============================================================
Maintains a 12-hour rolling deque buffer (144 samples @ 5-min cadence),
computes 57 dynamic solar wind features, sends payloads to FastAPI `/predict`,
and persists output into PostgreSQL / SQLite.
"""

import json
import os
import time
import requests
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional
from db_manager import DatabaseManager

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
WINDOW_STEPS_12H = 144  # 144 samples @ 5-min cadence = 12 Hours

BASE_PARAMS = [
    "plasma_speed_kmps",
    "B_magnitude",
    "proton_density",
    "bz_gse",
    "flow_pressure_npa",
    "plasma_temperature_k",
]

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
META_PATH = ARTIFACT_DIR / "model_meta.json"

with open(META_PATH, "r") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]


class TelemetryStreamWorker:
    def __init__(self, window_size: int = WINDOW_STEPS_12H, api_url: str = API_URL):
        self.window_size = window_size
        self.api_url = api_url
        self.buffer = deque(maxlen=window_size)
        self.db = DatabaseManager()
        print(f"[Phase 3 Worker] Sliding deque buffer initialized with capacity = {window_size} samples (12 Hours).")

    def compute_rolling_features(self, current_record: Dict[str, float]) -> Dict[str, float]:
        self.buffer.append(current_record)
        df_win = pd.DataFrame(list(self.buffer))

        engineered = {}
        for col in FEATURE_NAMES:
            if col in current_record:
                engineered[col] = current_record[col]

        n_samples = len(df_win)
        step_3h = min(n_samples, 36)
        step_6h = min(n_samples, 72)
        step_12h = min(n_samples, 144)
        step_1h = min(n_samples, 12)

        for param in BASE_PARAMS:
            if param in df_win.columns:
                series = df_win[param]

                # Rolling Means
                engineered[f"{param}_mean3h"] = float(series.iloc[-step_3h:].mean())
                engineered[f"{param}_mean6h"] = float(series.iloc[-step_6h:].mean())
                engineered[f"{param}_mean12h"] = float(series.iloc[-step_12h:].mean())

                # Rolling Standard Deviations
                engineered[f"{param}_std3h"] = float(series.iloc[-step_3h:].std()) if step_3h > 1 else 0.0
                engineered[f"{param}_std6h"] = float(series.iloc[-step_6h:].std()) if step_6h > 1 else 0.0
                engineered[f"{param}_std12h"] = float(series.iloc[-step_12h:].std()) if step_12h > 1 else 0.0

                # 1-hour delta
                val_now = float(series.iloc[-1])
                val_1h_ago = float(series.iloc[-step_1h]) if n_samples >= step_1h else val_now
                engineered[f"{param}_delta1h"] = float(val_now - val_1h_ago)

        final_features = {}
        for col in FEATURE_NAMES:
            final_features[col] = float(engineered.get(col, current_record.get(col, 0.0)))

        return final_features

    def process_telemetry_packet(self, packet: Dict[str, Any], invoke_api: bool = True) -> Dict[str, Any]:
        timestamp = packet.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        raw_telemetry = packet.get("telemetry", packet)

        engineered_features = self.compute_rolling_features(raw_telemetry)
        prediction_result = None

        if invoke_api:
            payload = {
                "timestamp": timestamp,
                "features": engineered_features,
            }
            try:
                resp = requests.post(self.api_url, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    prediction_result = resp.json()
                else:
                    print(f"[Worker Warning] API status code {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[Worker Error] API connection failed: {e}")

        return {
            "timestamp": timestamp,
            "engineered_features_count": len(engineered_features),
            "prediction": prediction_result,
        }


if __name__ == "__main__":
    worker = TelemetryStreamWorker()
    print("[StreamWorker] Standalone daemon ready.")
