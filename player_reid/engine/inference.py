import cv2
import torch
import numpy as np
import argparse
import os
import yaml
from ultralytics import YOLO
from player_reid.models.backbone import create_osnet
from player_reid.datasets.transforms import build_transforms

class PlayerInferenceEngine:
    def __init__(self, config_path, weights_path, device='cpu'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.device = torch.device(device)
        self.model = create_osnet(num_classes=self.cfg['MODEL']['NUM_CLASSES'])
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = build_transforms(is_train=False)
        self.det_model = YOLO("yolov8n.pt") # Standard YOLOv8n for demo

    def process_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

    def run_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Simple player detection and ReID labeling for demo
            results = self.det_model(frame, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls) == 0: # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        feat = self.process_image(crop)
                        # ID labeling logic would go here in full system
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            writer.write(frame)
            
        cap.release()
        writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player ReID Inference Engine")
    parser.add_argument("--config", type=str, default="player_reid/configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples/output_demo.mp4")
    args = parser.parse_args()
    
    engine = PlayerInferenceEngine(args.config, args.weights)
    engine.run_video(args.video, args.output)
