"""
Phase 3 & 4 E2E Integration Test
=================================
Usage:
    python Project-Code/ML_Model/test_phase3_4.py
"""

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app import app
from stream_worker import TelemetryStreamWorker
from db_manager import DatabaseManager
import json

client = TestClient(app)

class LocalAPIWorker(TelemetryStreamWorker):
    """Subclass of TelemetryStreamWorker that invokes FastAPI directly via TestClient."""

    def process_telemetry_packet(self, packet: dict, invoke_api: bool = True):
        timestamp = packet.get("timestamp", "2026-09-05T00:00:00Z")
        raw_telemetry = packet.get("telemetry", packet)

        engineered_features = self.compute_rolling_features(raw_telemetry)

        # Direct invocation via TestClient
        payload = {"timestamp": timestamp, "features": engineered_features}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"API failed with code {resp.status_code}"
        prediction_result = resp.json()

        return {
            "timestamp": timestamp,
            "engineered_features_count": len(engineered_features),
            "prediction": prediction_result,
        }


def run_e2e_tests():
    print("=" * 65)
    print("  PHASE 3 & 4: END-TO-END INTEGRATION VERIFICATION")
    print("=" * 65)

    worker = LocalAPIWorker(window_size=36)

    # 1. Simulate 10 sequential telemetry packets
    print("\n[1/3] Streaming 10 telemetry packets into Sliding Window Engine...")
    base_telemetry = {
        "bx_gse": -1.1, "by_gse": 1.9, "bz_gse": -1.5, "by_gsm": 1.6, "bz_gsm": 0.1,
        "proton_density": 365.0, "plasma_speed_kmps": -1.3, "flow_pressure_npa": -1.2,
        "plasma_temperature_k": 1.17, "B_magnitude": 2.65, "alfven_speed": 3.03,
        "dynamic_pressure": 0.001, "bz_southward": 1.5, "mach_alfven": -0.42, "swis_total_counts": 0.0
    }

    for i in range(10):
        t_data = base_telemetry.copy()
        t_data["plasma_speed_kmps"] += i * 10.0
        t_data["proton_density"] += i * 5.0

        ts = f"2026-09-05T18:{i:02d}:00Z"
        res = worker.process_telemetry_packet({"timestamp": ts, "telemetry": t_data})
        pred = res["prediction"]
        print(f"      Step {i+1:02d}: Prob={pred['cme_probability']:.4f} | Risk={pred['risk_level']:<8} | Halo={pred['halo_cme_predicted']}")

    # 2. Verify Database Persistence
    print("\n[2/3] Verifying Database Storage...")
    db_records = worker.db.get_historical_predictions(limit=10)
    print(f"      Retrieved {len(db_records)} records from Database.")
    assert len(db_records) >= 10, "Database records count mismatch!"

    # 3. Verify Database Record Integrity
    latest = db_records[0]
    print(f"      Latest DB Record -> Timestamp: {latest['timestamp']} | Risk: {latest['risk_level']} | Prob: {latest['cme_probability']}")
    assert "risk_level" in latest, "Missing risk_level in DB record!"
    assert "cme_probability" in latest, "Missing cme_probability in DB record!"

    print("\n" + "=" * 65)
    print("  PHASE 3 & 4 COMPLETE: STREAMING & STORAGE VERIFIED 100% CLEAN")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_tests()
