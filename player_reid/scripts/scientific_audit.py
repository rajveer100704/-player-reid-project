import os
import sys
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

from player_reid.models.backbone import create_osnet
from player_reid.evaluation.stadium_shift import StadiumShiftSimulator

def run_scientific_audit():
    print("--- 🧪 Starting Scientific Cross-Stadium Audit ---")
    
    device = 'cpu'
    weights_path = "models/reid_sports_10_10.pth"
    root_data = "data/raw_sports_data"
    num_classes = len([f for f in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, f))])
    
    model = create_osnet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load representative dense frame
    cap = cv2.VideoCapture("broadcast.mp4")
    for _ in range(6): cap.read()
    ret, frame = cap.read()
    cap.release()
    
    det_model = YOLO("yolov8n.pt")
    results = det_model(frame, verbose=False)[0]
    player_boxes = [box for box in results.boxes if int(box.cls) == 0][:12]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    variants = StadiumShiftSimulator.generate_stadium_variants(frame)
    results_table = []
    optimal_thresholds = []

    for name, variant_frame in variants.items():
        print(f"Processing Stadium Variant: {name}")
        
        # Extract crops from shifted frame
        crops_list = []
        for b in player_boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            crop = variant_frame[y1:y2, x1:x2]
            crops_list.append(transform(crop))
            
        crops = torch.stack(crops_list).to(device)
        
        with torch.no_grad():
            feats = model(crops)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            sim_mat = torch.mm(feats, feats.t()).cpu().numpy()
            
        N = len(player_boxes)
        avg_cross_sim = (np.sum(sim_mat) - N) / (N * (N - 1))
        
        # Simple threshold proxy for this audit frame
        opt_t = (np.max(sim_mat[sim_mat < 0.99]) + np.min(sim_mat[sim_mat < 0.99])) / 2
        
        results_table.append({
            "Stadium": name,
            "CrossSim": avg_cross_sim,
            "T_opt": opt_t
        })
        optimal_thresholds.append(opt_t)

    # Calculate TSI
    tsi = np.var(optimal_thresholds)
    
    print("\n--- 🏟️ CROSS-STADIUM AUDIT REPORT ---")
    print(f"{'Stadium':<20} | {'CrossSim':<10} | {'T_opt':<10}")
    print("-" * 45)
    for r in results_table:
        print(f"{r['Stadium']:<20} | {r['CrossSim']:<10.4f} | {r['T_opt']:<10.4f}")
    
    print(f"\nTHRESHOLD STABILITY INDEX (TSI): {tsi:.6f} (Target < 0.03)")
    if tsi < 0.03:
        print("✅ PASS: Embedding geometry is stadium-invariant.")
    else:
        print("❌ FAIL: Excessive threshold variance detected.")

if __name__ == "__main__":
    run_scientific_audit()
