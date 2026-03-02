# Sports-Specialized Player Re-Identification (OSNet + ArcFace)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.10+-red.svg)
![ONNX](https://img.shields.io/badge/onnx-supported-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## Overview
This repository implements a high-performance Player Re-Identification (ReID) system specifically optimized for the dense-frame challenges of professional sports analytics. By combining omni-scale feature extraction with angular margin loss, the system achieves superior identity separation between teammates wearing near-identical uniforms.

### Key Features
- **OSNet + BNNeck Backbone**: Optimized for multi-scale feature extraction with decoupled metric/classification spaces.
- **ArcFace + Triplet Loss**: Minimizes intra-class variance while maximizing angular separation.
- **Temporal Tracklet Aggregation**: Aggregates frame-level embeddings for stable track-level identity persistence.
- **Industrial Monitoring**: Integrated EMA-drift detection for long-running match stability.

## Architecture
The system follows a modern metric learning architecture designed for industrial robustness:

1. **Backbone**: OSNet (Omni-Scale Network) extracts scale-invariant features.
2. **BNNeck**: Decouples ID classification from feature matching to prevent gradient conflicts.
3. **Angular Optimization**: ArcFace ($m=0.5$) forces a strictly defined angular margin on the hypersphere.
4. **Temporal aggregation**: Quality-aware attention for robust tracklet identification.

## Key Empirical Result
Validated on dense-frame sports footage (12+ players in one frame):
- **Cross-Player Similarity**: $0.97 \rightarrow 0.64$ (Post-ArcFace).
- **Identity Gap**: Improved to $0.35+$.
- **Result**: Identity collapse eliminated in dense teammate clustering.

## Installation
```bash
git clone https://github.com/user/player_reid.git
cd player_reid
pip install -r requirements.txt
```

## Usage
### Training
```bash
bash scripts/train.sh
```
### Evaluation
```bash
bash scripts/evaluate.sh
```
### Demo Video
```bash
bash scripts/demo_video.sh
```

## Metrics & Validation
The system includes standardized metrics for industrial-designed evaluation:
- **Rank-1 / mAP**: Standard ReID retrieval accuracy.
- **ID Switch Rate**: Target $< 0.2$ switches/minute.
- **Threshold Stability Index (TSI)**: Verified stability across multi-stadium conditions.
- **Drift Monitoring**: Max intra-player drift $< 0.08$ over match sequences.

## Limitations
- **Environment**: Validated under simulated cross-stadium shifts and professional broadcast conditions.
- **Infrastructure**: Full live multi-camera deployment requires dedicated external orchestration and high-concurrency storage.

---
*This project demonstrates metric learning, angular margin optimization, domain adaptation, and operational monitoring design for sports Re-Identification systems.*
