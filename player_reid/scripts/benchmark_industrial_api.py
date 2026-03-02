import asyncio
import httpx
import time
import numpy as np
import cv2
import io

async def benchmark_endpoint(url, num_requests=50, concurrency=5):
    # Prepare dummy image
    img = np.zeros((256, 128, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    latencies = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Initial warm-up
        await client.post(url, files={"file": ("dummy.jpg", img_bytes)})
        
        # Concurrent execution
        tasks = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request():
            async with semaphore:
                start = time.time()
                resp = await client.post(url, files={"file": ("test.jpg", img_bytes)})
                latencies.append(time.time() - start)
                return resp.status_code

        for _ in range(num_requests):
            tasks.append(make_request())
            
        print(f"Sending {num_requests} requests with concurrency {concurrency}...")
        start_total = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_total
        
    # Analyze
    latencies = np.array(latencies) * 1000 # to ms
    print(f"\n--- ⚡ Performance Report ---")
    print(f" Requests: {num_requests}")
    print(f" Concurrency: {concurrency}")
    print(f" Total Time: {total_time:.2f}s")
    print(f" Avg Latency: {np.mean(latencies):.2f}ms")
    print(f" p50 Latency: {np.percentile(latencies, 50):.2f}ms")
    print(f" p95 Latency: {np.percentile(latencies, 95):.2f}ms")
    print(f" p99 Latency: {np.percentile(latencies, 99):.2f}ms")
    print(f" Throughput: {num_requests / total_time:.2f} reID/sec")

if __name__ == "__main__":
    # Note: This requires the server running in background.
    # In a real environment, we'd start it. Here we simulate the logic.
    print("Pre-flight check: Ensure app_industrial.py is running on port 8000.")
    # asyncio.run(benchmark_endpoint("http://localhost:8000/extract_features"))
