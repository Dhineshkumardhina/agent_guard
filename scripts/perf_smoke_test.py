import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.app.main import app

def run_performance_smoke_test():
    client = TestClient(app)

    endpoints = [
        ("/health", "Service health check"),
        ("/api/v1/agents?limit=50", "Agents list (50 items)"),
        ("/api/v1/runs?limit=50", "Runs list (50 items)"),
        ("/api/v1/runs/run_0005_agentguard_generalization_v1", "Single run detail"),
        ("/api/v1/runs/run_0005_agentguard_generalization_v1/graph", "Temporal graph slice"),
        ("/api/v1/runs/run_telemetry_demo_001/events", "Event stream"),
        ("/api/v1/evaluations/comparison?horizon=1", "Canonical 9-model comparison"),
        ("/api/v1/ablations?limit=50", "Ablation list (50 items)"),
        ("/api/v1/generalization?limit=50", "Generalization list (50 items)"),
        ("/api/v1/explanations", "Explanations list"),
    ]

    print(f"{'Endpoint':55} | {'Latency (ms)':>12} | {'Status':>6}")
    print("-" * 78)
    for ep, desc in endpoints:
        # Warmup
        client.get(ep)
        # Measured runs (average of 3)
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = client.get(ep)
            timings.append((time.perf_counter() - t0) * 1000)
        avg_ms = sum(timings) / len(timings)
        print(f"{ep:55} | {avg_ms:12.2f} | {res.status_code:>6}")

if __name__ == "__main__":
    run_performance_smoke_test()
