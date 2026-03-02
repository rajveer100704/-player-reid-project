import cv2
import os
import sys
from ultralytics import YOLO
from pathlib import Path

def extract_tracklets(video_path, output_dir, model_path='yolov8n.pt', conf=0.5, iou=0.5):
    """
    Extracts player crops from a video, grouped by Track ID using YOLOv8 tracking.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_idx = 0
    print(f"Starting tracklet extraction for {video_path.name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking (bytetrack is default in YOLOv8)
        results = model.track(frame, persist=True, verbose=False, conf=conf, iou=iou)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, clss):
                # Filter for 'person' class (index 0 in COCO)
                if cls == 0:
                    x1, y1, x2, y2 = box
                    # Ensure coordinates are within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    y2, x2 = min(frame.shape[0], y2), min(frame.shape[1], x2)
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue

                    # Save crop
                    track_dir = output_dir / f"track_{video_path.stem}_{track_id}"
                    track_dir.mkdir(exist_ok=True)
                    
                    crop_filename = f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(track_dir / crop_filename), crop)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    print(f"Finished extraction. Data saved to {output_dir}")

if __name__ == "__main__":
    # Define paths
    # Note: Using absolute paths relative to execution dir
    videos = [
        "c:/Users/BIT/REID V1.0/-player-reid-project/broadcast.mp4",
        "c:/Users/BIT/REID V1.0/-player-reid-project/tacticam.mp4",
        "c:/Users/BIT/REID V1.0/-player-reid-project/15sec_input_720p.mp4"
    ]
    
    output_base = "c:/Users/BIT/REID V1.0/-player-reid-project/data/raw_sports_data"
    
    for video in videos:
        if os.path.exists(video):
            extract_tracklets(video, output_base)
        else:
            # Fallback for relative paths if run from root
            rel_video = os.path.basename(video)
            if os.path.exists(rel_video):
                extract_tracklets(rel_video, output_base)
            else:
                print(f"Skipping {video}, file not found.")
