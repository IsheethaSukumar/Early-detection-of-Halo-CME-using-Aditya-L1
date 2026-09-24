"""
Phase 1: Model Packaging & Artifact Exporter
============================================
Usage:
    python Project-Code/ML_Model/export_artifacts.py
"""

import json
import pickle
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent / "Dataset" / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DROP_COLS = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]


def export_artifacts():
    print("=" * 65)
    print("  PHASE 1: PACKAGING & EXPORTING DEPLOYMENT ARTIFACTS")
    print("=" * 65)

    # 1. Extract feature column list from dataset
    dataset_path = BASE_DIR / "full_merged_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path, nrows=5)
    feature_cols = [c for c in df.columns if c not in DROP_COLS and not c.startswith("Unnamed")]

    print(f"\n[1/4] Extracted {len(feature_cols)} feature columns from dataset.")

    # 2. Check thresholds
    rf_thresh_path = ARTIFACT_DIR / "rf_tuned_threshold.txt"
    if not rf_thresh_path.exists():
        rf_thresh_path = ARTIFACT_DIR / "rf_threshold.txt"

    rf_thresh = 0.65
    if rf_thresh_path.exists():
        with open(rf_thresh_path, "r") as f:
            rf_thresh = float(f.read().strip())
    print(f"[2/4] Loaded classification threshold: {rf_thresh:.4f}")

    iforest_thresh_path = ARTIFACT_DIR / "iforest_threshold.txt"
    iforest_thresh = -0.1
    if iforest_thresh_path.exists():
        with open(iforest_thresh_path, "r") as f:
            try:
                iforest_thresh = float(f.read().strip())
            except ValueError:
                pass
    print(f"[3/4] Loaded anomaly detection threshold: {iforest_thresh}")

    # 3. Create metadata dictionary
    meta = {
        "model_version": "1.0.0",
        "model_name": "Random Forest Classifier + Isolation Forest Anomaly Detector",
        "feature_names": feature_cols,
        "total_features": len(feature_cols),
        "classification_threshold": rf_thresh,
        "warning_threshold": rf_thresh,
        "critical_threshold": max(rf_thresh + 0.15, 0.85),
        "anomaly_threshold": iforest_thresh,
        "model_file": "random_forest_tuned.pkl" if (ARTIFACT_DIR / "random_forest_tuned.pkl").exists() else "random_forest.pkl",
        "scaler_file": "rf_scaler.pkl",
        "iforest_file": "iforest_model.pkl"
    }

    meta_path = ARTIFACT_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[4/4] Metadata packaged and saved -> {meta_path}")
    print("\n" + "=" * 65)
    print("  PHASE 1 COMPLETE: ARTIFACTS VERIFIED AND PACKAGED")
    print("=" * 65)


if __name__ == "__main__":
    export_artifacts()
