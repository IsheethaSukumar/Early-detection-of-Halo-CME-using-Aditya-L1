"""
Phase 2 API Verification Test
==============================
Usage:
    python Project-Code/ML_Model/test_phase2_api.py
"""

from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def run_tests():
    print("=" * 65)
    print("  PHASE 2: FASTAPI MICROSERVICE END-TO-END VERIFICATION")
    print("=" * 65)

    # 1. Test /health
    res_health = client.get("/health")
    print(f"\n[1/4] Testing GET /health -> Status Code: {res_health.status_code}")
    print(f"      Response: {res_health.json()}")
    assert res_health.status_code == 200, "Health check failed!"

    # 2. Test /features
    res_feats = client.get("/features")
    print(f"\n[2/4] Testing GET /features -> Total Features Received: {res_feats.json()['total_features']}")
    assert res_feats.status_code == 200, "Features endpoint failed!"

    # 3. Test /predict/sample
    res_sample = client.get("/predict/sample")
    print(f"\n[3/4] Testing GET /predict/sample -> Status Code: {res_sample.status_code}")
    print(f"      Sample Prediction Output:\n{json.dumps(res_sample.json(), indent=6)}")
    assert res_sample.status_code == 200, "Sample endpoint failed!"

    # 4. Test /predict (Single Payload)
    features_meta = res_feats.json()["feature_names"]
    test_payload = {
        "timestamp": "2026-09-05T18:15:00Z",
        "features": {col: 1.5 for col in features_meta}
    }
    res_pred = client.post("/predict", json=test_payload)
    print(f"\n[4/4] Testing POST /predict -> Status Code: {res_pred.status_code}")
    print(f"      Single Inference Result:\n{json.dumps(res_pred.json(), indent=6)}")
    assert res_pred.status_code == 200, "Predict endpoint failed!"

    print("\n" + "=" * 65)
    print("  PHASE 2 VERIFICATION COMPLETE: ALL API ENDPOINTS WORKING PERFECTLY")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
