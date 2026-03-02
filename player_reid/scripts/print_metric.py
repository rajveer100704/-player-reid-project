import torch
import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())
from player_reid.models.backbone import create_osnet

def get_10_10_metrics():
    device = 'cpu'
    weights_path = "models/reid_sports_10_10.pth"
    root_data = "data/raw_sports_data"
    num_classes = len([f for f in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, f))])
    
    model = create_osnet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture("broadcast.mp4")
    # Frame 6
    for _ in range(6): cap.read()
    ret, frame = cap.read()
    cap.release()
    
    det_model = YOLO("yolov8n.pt")
    results = det_model(frame, verbose=False)[0]
    
    player_boxes = results.boxes[:12]
    N = len(player_boxes)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    crops_list = []
    for b in player_boxes:
        coords = b.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        crop = frame[y1:y2, x1:x2]
        crops_list.append(transform(crop))
        
    crops = torch.stack(crops_list).to(device)
    
    with torch.no_grad():
        feats = model(crops)
        # Apply L2 normalization
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        sim_mat = torch.mm(feats, feats.t()).cpu().numpy()
    
    avg_cross_sim = (np.sum(sim_mat) - N) / (N * (N - 1))
    print(f"METRIC_VALUE_SIM: {avg_cross_sim:.4f}")

if __name__ == "__main__":
    get_10_10_metrics()
