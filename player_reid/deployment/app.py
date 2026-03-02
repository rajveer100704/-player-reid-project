from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
from player_reid.models.backbone import create_osnet
from player_reid.datasets.transforms import build_transforms

app = FastAPI(title="Professional Player ReID API")

# Load model (Global for performance)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_osnet()
# model.load_state_dict(torch.load("path/to/best_model.pth")['state_dict'])
model.to(device)
model.eval()

transform = build_transforms(is_train=False)

@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        features = model(input_tensor)
    
    return {"features": features.cpu().numpy().tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
