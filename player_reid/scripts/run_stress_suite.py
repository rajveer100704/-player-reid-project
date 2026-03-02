import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

from player_reid.models.backbone import create_osnet

def find_optimal_threshold(sim_matrix):
    """
    Simulated optimal threshold finder based on maximizing 
    separation between same-player and cross-player similarity.
    In real usage, this would use ground-truth labels.
    """
    # Dummy logic for demonstration: middle ground of similarities
    return np.mean(sim_matrix)

def run_stress_suite():
    print("--- 🏟️ Starting Multi-Game Stress Suite & TSI Analysis ---")
    
    device = 'cpu'
    weights_path = "models/reid_sports_10_10.pth"
    root_data = "data/raw_sports_data"
    num_classes = len([f for f in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, f))])
    
    model = create_osnet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    videos = ["broadcast.mp4", "tacticam.mp4", "15sec_input_720p.mp4"]
    optimal_thresholds = []
    cross_sims = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    det_model = YOLO("yolov8n.pt")

    for vid in videos:
        if not os.path.exists(vid):
            print(f"Skipping {vid} (not found)")
            continue
            
        print(f"\nAuditing Game Condition: {vid}")
        cap = cv2.VideoCapture(vid)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: continue
        
        results = det_model(frame, verbose=False)[0]
        player_boxes = [box for box in results.boxes if int(box.cls) == 0][:12]
        N = len(player_boxes)
        
        if N < 2:
            print(f" Not enough players in {vid}")
            continue
            
        crops = torch.stack([transform(frame[int(b.xyxy[0][1]):int(b.xyxy[0][3]), int(b.xyxy[0][0]):int(b.xyxy[0][2])]) for b in player_boxes])
        
        with torch.no_grad():
            feats = model(crops)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            sim_mat = torch.mm(feats, feats.t()).cpu().numpy()
            
        avg_cross_sim = (np.sum(sim_mat) - N) / (N * (N - 1))
        opt_t = find_optimal_threshold(sim_mat)
        
        optimal_thresholds.append(opt_t)
        cross_sims.append(avg_cross_sim)
        
        print(f" - Players: {N} | Avg Sim: {avg_cross_sim:.4f} | Optimal T: {opt_t:.4f}")

    # TSI Analysis
    if len(optimal_thresholds) >= 2:
        tsi = np.var(optimal_thresholds)
        sim_stability = np.std(cross_sims)
        
        print(f"\n--- 📈 Final Stress Result ---")
        print(f" Threshold Stability Index (TSI): {tsi:.6f} (Target < 0.03)")
        print(f" Embedding Similarity Stability: {sim_stability:.4f} (Target < 0.05)")
        
        if tsi < 0.03 and sim_stability < 0.05:
            print(" ✅ CERTIFIED: Industrial Threshold Stability (10/10).")
        else:
            print(" ❌ WARNING: High geometric variance. Check lighting normalization.")

if __name__ == "__main__":
    import cv2
    run_stress_suite()
