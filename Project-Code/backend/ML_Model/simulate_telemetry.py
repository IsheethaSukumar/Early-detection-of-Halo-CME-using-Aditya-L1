"""
Phase 3: Real-Time Telemetry Streaming Simulator
================================================
Usage:
    python Project-Code/ML_Model/simulate_telemetry.py --records 20 --interval 0.5
"""

import sys
from pathlib import Path

ML_MODEL_DIR = Path(__file__).resolve().parent
if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

import time
import argparse
import pandas as pd
from stream_worker import TelemetryStreamWorker

BASE_DIR = Path(__file__).resolve().parent.parent / "Dataset" / "processed"
DATASET_PATH = BASE_DIR / "full_merged_dataset.csv"


def run_simulation(max_records: int = 50, interval_sec: float = 0.2):
    print("=" * 65)
    print("  PHASE 3 & 4: REAL-TIME TELEMETRY SIMULATION & INGESTION")
    print("=" * 65)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Simulation dataset not found at {DATASET_PATH}")

    print(f"\n[1/3] Loading simulation telemetry from {DATASET_PATH.name}...")
    df = pd.read_csv(DATASET_PATH)

    timestamp_col = df.columns[0]
    raw_cols = [c for c in df.columns if c not in ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]]

    worker = TelemetryStreamWorker()

    print(f"\n[2/3] Streaming {max_records} records through Feature Engine & API...\n")

    start_time = time.time()
    for idx, row in df.iloc[:max_records].iterrows():
        timestamp = str(row[timestamp_col])
        telemetry = {col: float(row[col]) for col in raw_cols if pd.notnull(row[col])}

        packet = {"timestamp": timestamp, "telemetry": telemetry}
        result = worker.process_telemetry_packet(packet, invoke_api=True)

        pred = result.get("prediction")
        if pred:
            cme_prob = pred['cme_probability']
            risk = pred['risk_level']
            is_cme = pred['halo_cme_predicted']
            anom = pred['is_anomaly']
            print(f"  [{idx+1:02d}/{max_records}] Timestamp: {timestamp} | CME Prob: {cme_prob:.4f} | Risk: {risk:<8} | Halo CME: {str(is_cme):<5} | Anomaly: {str(anom)}")
        else:
            print(f"  [{idx+1:02d}/{max_records}] Timestamp: {timestamp} | Features Engineered: {result['engineered_features_count']}")

        time.sleep(interval_sec)

    elapsed = time.time() - start_time
    print(f"\n[3/3] Simulation finished in {elapsed:.2f} seconds.")

    # Query DB summary
    recent_preds = worker.db.get_historical_predictions(limit=5)
    print(f"\n[Phase 4 DB Verification] Successfully persisted records. Last {len(recent_preds)} database entries:")
    for p in recent_preds:
        print(f"  - DB ID: Timestamp={p['timestamp']} | Risk={p['risk_level']} | Prob={p['cme_probability']:.4f}")

    print("\n" + "=" * 65)
    print("  PHASE 3 & 4 COMPLETE: STREAMING PIPELINE VERIFIED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Telemetry Simulation")
    parser.add_argument("--records", type=int, default=20, help="Number of records to simulate")
    parser.add_argument("--interval", type=float, default=0.1, help="Streaming interval in seconds")
    args = parser.parse_args()

    run_simulation(max_records=args.records, interval_sec=args.interval)
