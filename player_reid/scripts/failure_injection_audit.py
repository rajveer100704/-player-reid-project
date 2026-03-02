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

def inject_noise(image, intensity=0.1):
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def inject_blur(image, intensity=5):
    if intensity % 2 == 0: intensity += 1
    return cv2.GaussianBlur(image, (intensity, intensity), 0)

def run_degradation_audit():
    print("--- 📉 Starting Failure Injection & Graceful Degradation Audit ---")
    
    device = 'cpu'
    weights_path = "models/reid_sports_10_10.pth"
    root_data = "data/raw_sports_data"
    num_classes = len([f for f in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, f))])
    
    model = create_osnet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

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

    corruption_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3] # Noise levels
    results_list = []

    for level in corruption_levels:
        corrupted_frame = inject_noise(frame, level)
        # Also add some blur proportional to level
        if level > 0:
            corrupted_frame = inject_blur(corrupted_frame, int(level * 30))
            
        crops_list = []
        for b in player_boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            crop = corrupted_frame[y1:y2, x1:x2]
            crops_list.append(transform(crop))
            
        crops = torch.stack(crops_list).to(device)
        
        with torch.no_grad():
            feats = model(crops)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            sim_mat = torch.mm(feats, feats.t()).cpu().numpy()
            
        N = len(player_boxes)
        avg_cross_sim = (np.sum(sim_mat) - N) / (N * (N - 1))
        
        # Identity Separation Confidence (1.0 - CrossSim)
        separation_confidence = 1.0 - avg_cross_sim
        results_list.append((level, separation_confidence))
        print(f" Corruption {level:.2f} | Separation Confidence: {separation_confidence:.4f}")

    print("\n--- 📈 DEGRADATION CURVE ---")
    print(f"{'Noise Level':<12} | {'Separation Confidence':<20}")
    print("-" * 35)
    for level, conf in results_list:
        print(f"{level:<12.2f} | {conf:<20.4f}")

    # Check for catastrophic collapse
    baseline_conf = results_list[0][1]
    final_conf = results_list[-1][1]
    drop = baseline_conf - final_conf
    
    print(f"\nTotal Confidence Drop: {drop:.4f}")
    if drop < 0.15:
        print("✅ PASS: Graceful degradation confirmed. System is resilient to environment noise.")
    else:
        print("❌ WARNING: Catastrophic collapse detected at high noise. Check robustness tuning.")

if __name__ == "__main__":
    run_degradation_audit()
