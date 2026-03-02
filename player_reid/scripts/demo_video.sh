#!/bin/bash
# Player ReID Demo Video Entry Point
python -m player_reid.engine.inference \
    --video player_reid/samples/sample_short.mp4 \
    --weights player_reid/models/reid_sports.pth \
    --output player_reid/samples/output_demo.mp4
