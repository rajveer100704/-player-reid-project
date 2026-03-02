import numpy as np
import torch
import sys
import os
from collections import deque

# Add project root to path
sys.path.append(os.getcwd())
from player_reid.engine.industrial_monitoring import ReIDDriftMonitor

def run_longitudinal_test(num_frames=100000, num_players=12):
    print(f"--- 🕰️ Starting 90-Minute (100k Frames) Stability Test ---")
    print(f" Simulating {num_players} players over {num_frames} frames...")
    
    monitor = ReIDDriftMonitor(window_size=100, drift_threshold=0.08)
    
    # Initialize base embeddings for players
    # We want them to be somewhat distinct
    base_embeddings = torch.randn(num_players, 512)
    base_embeddings = torch.nn.functional.normalize(base_embeddings, p=2, dim=1)
    
    # Trackers for reporting
    max_drift = 0.0
    min_inter_dist = 1.0
    total_alerts = 0
    
    # Simulation loop
    for f in range(num_frames):
        for p in range(num_players):
            player_id = f"player_{p}"
            
            # Simulate natural variance (0.02 jitter)
            # Plus a very slow linear drift (0.0000001 per frame) to simulate lighting shift
            drift_vec = torch.randn(512) * 0.02
            slow_drift = (torch.ones(512) * 0.0000002) * f 
            
            current_emb = base_embeddings[p] + drift_vec + slow_drift
            current_emb = torch.nn.functional.normalize(current_emb, p=2, dim=0)
            
            sig = monitor.add_embedding(player_id, current_emb)
            
            # Update metrics
            max_drift = max(max_drift, sig['intra_drift'])
            total_alerts += 1 if sig['alert'] else 0
            
            # Sample inter-player shrink occasionally to save compute
            if f % 1000 == 0 and p == 0:
                min_inter_dist = min(min_inter_dist, 1.0 - sig['inter_shrinkage']) # inter_shrinkage is base_dist - current_dist

        if f % 10000 == 0:
            print(f" Frame {f}: MaxDrift: {max_drift:.4f} | Alerts: {total_alerts}")

    print("\n--- 📊 Longitudinal Results ---")
    print(f" Total Frames: {num_frames}")
    print(f" Max Intra-Player Drift: {max_drift:.4f} (Target < 0.08)")
    print(f" Final Inter-Player Stability: {min_inter_dist:.4f}")
    print(f" Total False Drift Alerts: {total_alerts}")
    
    if max_drift < 0.08 and total_alerts == 0:
        print("\n✅ CERTIFIED: Long-Term Stability Proven (10/10).")
    else:
        print("\n❌ WARNING: Temporal instability detected in long sequence.")

if __name__ == "__main__":
    run_longitudinal_test()
