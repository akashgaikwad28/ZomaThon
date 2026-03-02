import time
import requests
import statistics
import random

# Configuration
API_URL = "http://localhost:8000/api/v1/recommend"
NUM_REQUESTS = 50

def run_benchmark():
    """
    Simulates real-time traffic and measures latency of the recommendation engine.
    """
    print(f"Starting latency benchmark on {API_URL}...")
    latencies = []
    
    for i in range(NUM_REQUESTS):
        payload = {
            "user_id": random.randint(1, 1000),
            "restaurant_id": random.randint(1, 100),
            "cart_items": random.sample(range(1, 500), random.randint(1, 3))
        }
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
            
    if not latencies:
        print("No successful requests to benchmark.")
        return

    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99_latency = max(latencies)
    
    print("\n--- Latency Benchmark Results ---")
    print(f"Mean Latency: {avg_latency:.2f}ms")
    print(f"P95 Latency:  {p95_latency:.2f}ms")
    print(f"P99 Latency:  {p99_latency:.2f}ms")
    print(f"Target SLA:   <250ms")
    
    if p99_latency < 250:
        print("RESULT: SUCCESS (Target met)")
    else:
        print("RESULT: FAILURE (SLA breached)")

if __name__ == "__main__":
    # Note: Requires the server to be running (python -m app.main)
    run_benchmark()
