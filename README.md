
This project aims to develop a solution for player re-identification in sports footage, addressing both cross-camera player mapping and re-identification in a single feed.

## Task Options:

### Option 1: Cross-Camera Player Mapping

**Objective:** Given two clips (broadcast.mp4 and tacticam.mp4) of the same gameplay from different camera angles, map the players such that each player retains a consistent ID across both feeds.

**Instructions:**
- Use the provided object detection model to detect players in both videos.
- Match each player from the tacticam video to their corresponding identity in the broadcast video using consistent player_id values.
- You may use any combination of visual, spatial, or temporal features to establish the mapping.

### Option 2: Re-Identification in a Single Feed

**Objective:** Given a 15-second video (15sec_input_720p.mp4), identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before.

**Instructions:**
- Use the provided object detection model to detect players throughout the clip.
- Assign player IDs based on the initial few seconds.
- Maintain the same ID for players when they re-enter the frame later in the video (e.g., near the goal event).
- Your solution should simulate real-time re-identification and player tracking.

## Model Download:

Object detection model link: [https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePvCMD/view](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePvCMD/view)

Note: The model is a basic fine-tuned version of Ultralytics YOLOv11, trained specifically on players and the ball.

## Submission Requirements:

Submit your work via a GitHub repository or a Google Drive folder, containing:

1. All source code used for the assignment.
2. A README.md or equivalent documentation clearly explaining:
    - How to set up and run the code.
    - Dependencies or environment requirements.
3. A brief report (PDF or Markdown) covering:
    - Your approach and methodology.
    - Techniques you tried and their outcomes.
    - Challenges encountered.
    - If incomplete, describe what remains and how you would proceed with more time/resources.

## Evaluation Criteria:

- Accuracy and reliability of player re-identification
- Simplicity, modularity, and clarity of code
- Documentation quality
- Runtime efficiency and latency (bonus points)
- Thoughtfulness and creativity of your approach

## Additional Notes:

This assignment is designed to reflect real-world computer vision constraints and open-ended problem-solving. A fully working solution is not mandatory. We are equally interested in how you extend this problem.
