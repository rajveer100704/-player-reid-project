# main.py

import cv2
from ultralytics import YOLO
import torch
from collections import defaultdict
import numpy as np
import os

# Define base directory for models and videos
BASE_DIR = "/home/ubuntu/player_reid_project"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models
object_detection_model = YOLO(os.path.join(MODELS_DIR, "object_detection_best.pt"))
reidentification_model = YOLO(os.path.join(MODELS_DIR, "reidentification_best.pt")) 

class PlayerTracker:
    def __init__(self):
        self.player_id_counter = 0
        self.player_features = {}
        self.player_locations = {}
        self.player_history = defaultdict(list) # To store historical features for better re-identification

    def get_next_id(self):
        self.player_id_counter += 1
        return self.player_id_counter

    def update_player(self, bbox, features):
        min_distance = float("inf")
        assigned_id = None

        # Compare with existing player features
        for player_id, stored_features_list in self.player_features.items():
            # Use the average or latest feature for comparison
            stored_features = stored_features_list[-1] # Using latest feature for simplicity
            distance = torch.norm(features - stored_features)
            if distance < min_distance and distance < 0.7:  # Threshold for re-identification
                min_distance = distance
                assigned_id = player_id
        
        if assigned_id is None:
            assigned_id = self.get_next_id()
            self.player_features[assigned_id] = [features] # Start new history
        else:
            self.player_features[assigned_id].append(features) # Append to existing history
        
        self.player_locations[assigned_id] = bbox # Update last known location
        return assigned_id

def extract_features(model, image_crop):
    # Assuming the reidentification_model takes an image crop and outputs features
    # This part needs to be adapted based on the actual reidentification_model architecture
    # For a YOLO-based feature extractor, you might need to access intermediate layers
    # For now, let\'s assume it directly outputs features when called with an image
    
    # Resize image_crop to a fixed size if required by the reidentification_model
    # For example, if the model expects 224x224 input:
    # resized_crop = cv2.resize(image_crop, (224, 224))
    # features = model(resized_crop).cpu().numpy() # Assuming model outputs numpy array

    # Placeholder: In a real scenario, you would pass the image_crop through the reidentification_model
    # For now, return a random tensor as a dummy feature
    return torch.rand(128) # Example: 128-dim feature vector

def process_video(video_path, output_path, scenario_type):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    player_tracker = PlayerTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object Detection
        results = object_detection_model(frame)
        annotated_frame = results[0].plot() # Draw bounding boxes and labels

        if scenario_type == "single_feed":
            for *xyxy, conf, cls in results[0].boxes.data:
                if object_detection_model.names[int(cls)] == "player":
                    player_bbox = [int(x) for x in xyxy] # x1, y1, x2, y2
                    
                    # Crop player image
                    x1, y1, x2, y2 = player_bbox
                    player_crop = frame[y1:y2, x1:x2]

                    if player_crop.shape[0] > 0 and player_crop.shape[1] > 0: # Ensure valid crop
                        features = extract_features(reidentification_model, player_crop)
                        player_id = player_tracker.update_player(player_bbox, features)
                        
                        # Draw player ID on the annotated frame
                        cv2.putText(annotated_frame, f"ID: {player_id}", (player_bbox[0], player_bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_path}")

def cross_camera_reidentification(broadcast_path, tacticam_path, output_broadcast_path, output_tacticam_path):
    cap_broadcast = cv2.VideoCapture(broadcast_path)
    cap_tacticam = cv2.VideoCapture(tacticam_path)

    if not cap_broadcast.isOpened():
        print(f"Error: Could not open broadcast video {broadcast_path}")
        return
    if not cap_tacticam.isOpened():
        print(f"Error: Could not open tacticam video {tacticam_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_broadcast = cv2.VideoWriter(output_broadcast_path, fourcc, cap_broadcast.get(cv2.CAP_PROP_FPS), 
                                    (int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    out_tacticam = cv2.VideoWriter(output_tacticam_path, fourcc, cap_tacticam.get(cv2.CAP_PROP_FPS), 
                                  (int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    player_tracker = PlayerTracker()

    while True:
        ret_broadcast, frame_broadcast = cap_broadcast.read()
        ret_tacticam, frame_tacticam = cap_tacticam.read()

        if not ret_broadcast and not ret_tacticam:
            break

        # Process broadcast frame
        if ret_broadcast:
            results_broadcast = object_detection_model(frame_broadcast)
            annotated_frame_broadcast = results_broadcast[0].plot()
            current_frame_players_broadcast = []
            for *xyxy, conf, cls in results_broadcast[0].boxes.data:
                if object_detection_model.names[int(cls)] == "player":
                    player_bbox = [int(x) for x in xyxy]
                    x1, y1, x2, y2 = player_bbox
                    player_crop = frame_broadcast[y1:y2, x1:x2]
                    if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
                        features = extract_features(reidentification_model, player_crop)
                        current_frame_players_broadcast.append({"bbox": player_bbox, "features": features, "id": None})
            
            # Process tacticam frame
            current_frame_players_tacticam = []
            if ret_tacticam:
                results_tacticam = object_detection_model(frame_tacticam)
                annotated_frame_tacticam = results_tacticam[0].plot()
                for *xyxy, conf, cls in results_tacticam[0].boxes.data:
                    if object_detection_model.names[int(cls)] == "player":
                        player_bbox = [int(x) for x in xyxy]
                        x1, y1, x2, y2 = player_bbox
                        player_crop = frame_tacticam[y1:y2, x1:x2]
                        if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
                            features = extract_features(reidentification_model, player_crop)
                            current_frame_players_tacticam.append({"bbox": player_bbox, "features": features, "id": None})

            # Cross-camera matching logic
            # Simple nearest neighbor matching between broadcast and tacticam players
            for b_player in current_frame_players_broadcast:
                min_distance = float("inf")
                matched_t_player = None
                for t_player in current_frame_players_tacticam:
                    distance = torch.norm(b_player["features"] - t_player["features"])
                    if distance < min_distance and distance < 0.8: # Adjust threshold as needed
                        min_distance = distance
                        matched_t_player = t_player
                
                if matched_t_player:
                    # Assign same ID to matched players
                    player_id = player_tracker.get_next_id()
                    b_player["id"] = player_id
                    matched_t_player["id"] = player_id
                    player_tracker.player_features[player_id] = [b_player["features"], matched_t_player["features"]]
                else:
                    # If no match, assign new ID or try to re-identify within its own stream
                    b_player["id"] = player_tracker.update_player(b_player["bbox"], b_player["features"])
            
            for t_player in current_frame_players_tacticam:
                if t_player["id"] is None:
                    t_player["id"] = player_tracker.update_player(t_player["bbox"], t_player["features"])

            # Annotate frames with IDs
            for player_data in current_frame_players_broadcast:
                if player_data["id"] is not None:
                    cv2.putText(annotated_frame_broadcast, f"ID: {player_data['id']}", 
                                (player_data["bbox"][0], player_data["bbox"][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out_broadcast.write(annotated_frame_broadcast)

            if ret_tacticam:
                for player_data in current_frame_players_tacticam:
                    if player_data["id"] is not None:
                        cv2.putText(annotated_frame_tacticam, f"ID: {player_data['id']}", 
                                    (player_data["bbox"][0], player_data["bbox"][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out_tacticam.write(annotated_frame_tacticam)

    cap_broadcast.release()
    cap_tacticam.release()
    out_broadcast.release()
    out_tacticam.release()
    cv2.destroyAllWindows()
    print(f"Processed broadcast video saved to {output_broadcast_path}")
    print(f"Processed tacticam video saved to {output_tacticam_path}")

if __name__ == "__main__":
    # Single-feed re-identification
    print("Processing 15sec_input_720p.mp4 for single-feed re-identification...")
    process_video(os.path.join(BASE_DIR, "15sec_input_720p.mp4"), os.path.join(BASE_DIR, "output_single_feed.mp4"), "single_feed")

    # Cross-camera player mapping
    print("Starting cross-camera re-identification...")
    cross_camera_reidentification(
        os.path.join(BASE_DIR, "broadcast.mp4"),
        os.path.join(BASE_DIR, "tacticam.mp4"),
        os.path.join(BASE_DIR, "output_broadcast.mp4"),
        os.path.join(BASE_DIR, "output_tacticam.mp4")
    )


