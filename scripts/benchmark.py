import asyncio
import aiohttp
import time
import json
import statistics

URL = "http://localhost:8002/api/v1/recommend"
NUM_REQUESTS = 10000
CONCURRENCY = 100

payload = {
    "user_id": 1,
    "restaurant_id": 1,
    "cart_items": [101]
}

async def fetch(session, sem):
    async with sem:
        start_time = time.perf_counter()
        try:
            async with session.post(URL, json=payload) as response:
                await response.read()
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000  # ms
        except Exception as e:
            print(f"Request failed: {e}")
            return None

async def run_benchmark():
    print(f"Starting benchmark: {NUM_REQUESTS} requests with concurrency {CONCURRENCY}...")
    sem = asyncio.Semaphore(CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, sem) for _ in range(NUM_REQUESTS)]
        
        start_global = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_global = time.perf_counter()
        
    latencies = [r for r in results if r is not None]
    failed = NUM_REQUESTS - len(latencies)
    
    if not latencies:
        print("All requests failed.")
        return

    latencies.sort()
    
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)
    throughput = len(latencies) / (end_global - start_global)
    
    report = {
        "total_requests": NUM_REQUESTS,
        "failed_requests": failed,
        "concurrency": CONCURRENCY,
        "throughput_req_per_sec": float(f"{throughput:.2f}"),
        "latency_ms": {
            "average": float(f"{avg:.2f}"),
            "p50": float(f"{p50:.2f}"),
            "p95": float(f"{p95:.2f}"),
            "p99": float(f"{p99:.2f}")
        }
    }
    
    print("\n=== Benchmark Results ===")
    print(json.dumps(report, indent=4))
    
    if p99 < 250:
        print("\n[OK] PASSED: P99 Latency is under 250ms.")
    else:
        print("\n[X] WARNING: P99 Latency exceeded 250ms goal.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
