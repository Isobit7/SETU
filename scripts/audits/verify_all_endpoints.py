"""
Comprehensive Pre-Launch Verification Script for SETU
Tests all REST endpoints, WebSocket streaming, model readiness, and latency in-process using FastAPI TestClient.
"""

import json
import time
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app.main import app

def run_preflight_checks():
    results = []
    print("\n" + "="*70)
    print("SETU PRE-LAUNCH SYSTEM AUDIT & ENDPOINT VERIFICATION")
    print("="*70 + "\n")

    with TestClient(app) as client:
        # Helper tester
        def check(name, method, path, **kwargs):
            t0 = time.perf_counter()
            try:
                if method == "GET":
                    resp = client.get(path, **kwargs)
                elif method == "POST":
                    resp = client.post(path, **kwargs)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                is_ok = 200 <= resp.status_code < 300
                res_obj = {
                    "name": name,
                    "method": method,
                    "path": path,
                    "status": "PASS" if is_ok else "FAIL",
                    "code": resp.status_code,
                    "ms": elapsed_ms,
                    "data": resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text
                }
                print(f"[{'PASS' if is_ok else 'FAIL'}] {name:<35} | {method} {path:<32} | {resp.status_code} ({elapsed_ms}ms)")
                return res_obj
            except Exception as e:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                res_obj = {
                    "name": name,
                    "method": method,
                    "path": path,
                    "status": "FAIL",
                    "error": str(e),
                    "ms": elapsed_ms
                }
                print(f"[FAIL] {name:<35} | {method} {path:<32} | ERROR: {e}")
                return res_obj

        # 1. Core Service Readiness & Health
        results.append(check("Root Endpoint", "GET", "/"))
        results.append(check("Health Check", "GET", "/health"))
        results.append(check("Health Check Alias", "GET", "/api/health"))

        # 2. Models Catalog & Individual Model Statuses
        results.append(check("Models Catalog", "GET", "/api/v1/models"))
        results.append(check("Sequence LSTM Status", "GET", "/api/v1/models/sequence_lstm/status"))
        results.append(check("RealSign Model Status", "GET", "/api/v1/models/realsign/status"))
        results.append(check("RealSign Classes (26 A-Z)", "GET", "/api/v1/models/realsign/classes"))
        results.append(check("INCLUDE Model Status", "GET", "/api/v1/models/include/status"))
        results.append(check("INCLUDE Classes (263)", "GET", "/api/v1/models/include/classes"))

        # 3. Vocabulary
        results.append(check("Vocabulary & Descriptions", "GET", "/api/v1/vocabulary"))

        # 4. Custom Dataset Samples Listing
        results.append(check("Custom Dataset Samples", "GET", "/api/v1/dataset/custom_samples"))

        # 5. REST Predict Letter (RealSign Inference)
        mock_landmarks = [{"x": 0.5, "y": 0.5 + i * 0.01, "z": 0.0} for i in range(21)]
        results.append(check(
            "Letter Prediction (Landmarks)",
            "POST",
            "/api/predict",
            json={"landmarks": mock_landmarks, "dominant_hand": "right"}
        ))

        # 6. REST Record Sample Endpoint (30 frames x 164D)
        test_seq = [[0.02] * 164 for _ in range(30)]
        results.append(check(
            "Record Sample (30x164D)",
            "POST",
            "/api/v1/dataset/record_sample",
            json={"label": "test_preflight_check", "sequence_164d": test_seq}
        ))

        # Clean up temporary preflight file if created
        from pathlib import Path
        for f in Path("data/processed").glob("custom_test_preflight_check_*.npz"):
            try:
                f.unlink()
            except Exception:
                pass

        # 7. WebSocket Live Streaming & Model Dynamic Switching
        ws_test_name = "WebSocket Live Recognition Stream"
        t0_ws = time.perf_counter()
        try:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, encoded = cv2.imencode(".jpg", blank)
            jpeg_bytes = encoded.tobytes()

            with client.websocket_connect("/api/v1/ws/recognize?model=realsign") as ws:
                # Send frame under realsign
                ws.send_bytes(jpeg_bytes)
                resp1 = ws.receive_json()
                assert resp1["type"] == "prediction"
                assert resp1["active_model"] == "realsign"

                # Switch to isl_sequence_lstm
                ws.send_text('{"action": "set_model", "model": "isl_sequence_lstm"}')
                info1 = ws.receive_json()
                assert info1["type"] == "info"

                # Send frame under isl_sequence_lstm
                ws.send_bytes(jpeg_bytes)
                resp2 = ws.receive_json()
                assert resp2["type"] == "prediction"
                assert resp2["active_model"] == "isl_sequence_lstm"

                # Switch to isl_mvp
                ws.send_text('{"action": "set_model", "model": "isl_mvp"}')
                info2 = ws.receive_json()
                assert info2["type"] == "info"

                ws.send_bytes(jpeg_bytes)
                resp3 = ws.receive_json()
                assert resp3["type"] == "prediction"
                assert resp3["active_model"] == "isl_mvp"

            elapsed_ws = round((time.perf_counter() - t0_ws) * 1000, 2)
            results.append({
                "name": ws_test_name,
                "method": "WS",
                "path": "/api/v1/ws/recognize",
                "status": "PASS",
                "code": 101,
                "ms": elapsed_ws,
                "data": {"realsign": resp1["type"], "sequence_lstm": resp2["type"], "isl_mvp": resp3["type"]}
            })
            print(f"[PASS] {ws_test_name:<35} | WS   /api/v1/ws/recognize      | 101 ({elapsed_ws}ms)")
        except Exception as e:
            elapsed_ws = round((time.perf_counter() - t0_ws) * 1000, 2)
            results.append({
                "name": ws_test_name,
                "method": "WS",
                "path": "/api/v1/ws/recognize",
                "status": "FAIL",
                "error": str(e),
                "ms": elapsed_ws
            })
            print(f"[FAIL] {ws_test_name:<35} | WS   /api/v1/ws/recognize      | ERROR: {e}")

    print("\n" + "="*70)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"AUDIT SUMMARY: {passed}/{total} ENDPOINTS PASSED ({round(passed/total*100, 1)}%)")
    print("="*70 + "\n")
    return results

if __name__ == "__main__":
    run_preflight_checks()
