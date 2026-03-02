#!/bin/bash
# Player ReID Evaluation Entry Point
python -m player_reid.evaluation.metrics --weights player_reid/models/reid_sports.pth
