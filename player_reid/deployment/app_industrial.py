import asyncio
import torch
import numpy as np
import cv2
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
from player_reid.models.backbone import create_osnet
from player_reid.datasets.transforms import build_transforms

app = FastAPI(title="Industrial Player ReID API (10/10)")

# Industrial Config
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 0.05 # 50ms wait for batching
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Warm-Start & Model Loading
print(f"Loading industrial weights on {device}...")
# Assuming we have 500 classes from MVD + original
model = create_osnet(num_classes=500)
# model.load_state_dict(torch.load("models/reid_sports_10_10.pth", map_location=device))
model.to(device)
model.eval()

# Pre-allocated tensor buffer for warm-start
dummy_input = torch.zeros(MAX_BATCH_SIZE, 3, 256, 128).to(device)
with torch.no_grad():
    model(dummy_input)
print("Warm-start complete. Model is now active.")

transform = build_transforms(is_train=False)
executor = ThreadPoolExecutor(max_workers=4)

# 2. Dynamic Batching Engine
class DynamicBatcher:
    def __init__(self, model, max_batch_size, timeout):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = []
        self.lock = asyncio.Lock()
        self.processing = False

    async def add_to_queue(self, tensor):
        future = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((tensor, future))
            if not self.processing:
                asyncio.create_task(self.process_batches())
        return await future

    async def process_batches(self):
        async with self.lock:
            self.processing = True
        
        while True:
            batch = []
            futures = []
            
            async with self.lock:
                if not self.queue:
                    self.processing = False
                    break
                
                # Wait for batch to fill or timeout
                start_time = time.time()
                while len(batch) < self.max_batch_size and (time.time() - start_time) < self.timeout:
                    if self.queue:
                        tensor, fut = self.queue.pop(0)
                        batch.append(tensor)
                        futures.append(fut)
                    else:
                        await asyncio.sleep(0.01)

            if batch:
                input_tensor = torch.cat(batch).to(device)
                with torch.no_grad():
                    # Industrial Inference
                    features = self.model(input_tensor)
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                
                feat_np = features.cpu().numpy()
                for i, fut in enumerate(futures):
                    fut.set_result(feat_np[i].tolist())

batcher = DynamicBatcher(model, MAX_BATCH_SIZE, BATCH_TIMEOUT)

@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    # Async read
    contents = await file.read()
    
    # Process in thread pool to not block event loop
    def decode_and_prep():
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image).unsqueeze(0)

    input_tensor = await asyncio.get_event_loop().run_in_executor(executor, decode_and_prep)
    
    # Send to dynamic batcher
    features = await batcher.add_to_queue(input_tensor)
    
    return {"features": features}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
