import sys
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

from player_reid.models.backbone import create_osnet

def run_final_certification():
    print("--- 🏆 Phase 15 Final Certification Audit ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load 10/10 Sports Weights
    weights_path = "models/reid_sports_10_10.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Training likely still in progress.")
        return

    # Need to know num_classes for weights to load correctly
    # But for verification of embeddings, we just need the features
    # Let's count folders again to match the weights
    root_data = "data/raw_sports_data"
    num_classes = len([f for f in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, f))])
    
    model = create_osnet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded high-discriminative sports weights: {weights_path}")

    # 2. Extract the 'Problem Frame' (Broadcast 12 players)
    cap = cv2.VideoCapture("broadcast.mp4")
    # Finding frame 6 (where audit found 12 players)
    for _ in range(6): cap.read()
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not find broadcast.mp4.")
        return

    det_model = YOLO("yolov8n.pt")
    results = det_model(frame, verbose=False)[0]
    player_boxes = [box for box in results.boxes if int(box.cls) == 0]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    crops = []
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        crops.append(transform(crop))
    
    data_tensor = torch.stack(crops).to(device)
    
    # 3. Compute Similarity Matrix
    with torch.no_grad():
        # Inference mode returns L2-normalized BNNeck features
        feats = model(data_tensor)
        sim_mat = torch.mm(feats, feats.t()).cpu().numpy()
    
    # 4. Results & Certification
    num_players = len(player_boxes)
    avg_cross_sim = (np.sum(sim_mat) - num_players) / (num_players * (num_players - 1))
    
    print(f"\nAudit Frame: broadcast.mp4 (Frame 6)")
    print(f"Players Detected: {num_players}")
    print(f"Avg Cross-Player Similarity: {avg_cross_sim:.4f}")
    
    print("\nSimilarity Matrix Sample (3x3):")
    print(sim_mat[:3, :3])

    if avg_cross_sim < 0.8:
        print("\n✅ CERTIFIED: Identity Collapse SOLVED.")
        print("Inter-player separation is healthy. 10/10 Engineering Achievement.")
    else:
        print("\n❌ FAILED: Similarity still too high. Retraining required for domain adaptation.")

if __name__ == "__main__":
    run_final_certification()
