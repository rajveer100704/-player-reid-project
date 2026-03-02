import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.getcwd())

from player_reid.engine.industrial_monitoring import ReIDDriftMonitor

def simulate_industrial_monitoring():
    print("--- 🏭 Starting Industrial Drift Monitoring Simulation ---")
    
    # 1. Initialize Monitor
    # Drift threshold 0.05 for high sensitivity
    monitor = ReIDDriftMonitor(window_size=10, drift_threshold=0.05)
    
    # 2. Simulate Baseline (Normal Play)
    player_id = "p1_striker"
    print(f"Establishing baseline for {player_id}...")
    
    # Create a base embedding (random 512-dim normalized)
    base_emb = torch.randn(512)
    base_emb = torch.nn.functional.normalize(base_emb, p=2, dim=0)
    
    for i in range(10):
        # Adding slight noise to simulate natural movement
        noisy_emb = base_emb + torch.randn(512) * 0.01
        noisy_emb = torch.nn.functional.normalize(noisy_emb, p=2, dim=0)
        monitor.add_embedding(player_id, noisy_emb)
        
    print("Baseline established (Window 10 complete).")
    
    # 3. Simulate Normal Operation (Stable)
    print("\nStarting normal operation monitoring...")
    for i in range(5):
        noisy_emb = base_emb + torch.randn(512) * 0.02 # Slightly more noise
        noisy_emb = torch.nn.functional.normalize(noisy_emb, p=2, dim=0)
        sig = monitor.add_embedding(player_id, noisy_emb)
        print(f" Frame {i+1}: Intra: {sig['intra_drift']:.4f} | EMA: {sig['ema_drift']:.4f} | Alert: {sig['alert']}")

    # 4. Simulate Drift (Lighting change / Shaded area)
    print("\n⚠️ Simulating environmental drift (Intra-player shift)...")
    drift_vec = torch.ones(512) * 0.1
    drifted_emb_base = base_emb + drift_vec
    drifted_emb_base = torch.nn.functional.normalize(drifted_emb_base, p=2, dim=0)
    
    for i in range(10):
        noisy_emb = drifted_emb_base + torch.randn(512) * 0.01
        noisy_emb = torch.nn.functional.normalize(noisy_emb, p=2, dim=0)
        sig = monitor.add_embedding(player_id, noisy_emb)
        if sig['alert']:
            print(f" 🔥 ALERT: Drift Detected! Frame {i+1} | Intra: {sig['intra_drift']:.4f}")
        else:
            print(f" Frame {i+1}: Intra: {sig['intra_drift']:.4f}")

    # 5. Simulate Inter-player Centroid Shrinkage (Collapse Relapse)
    print("\n👥 Simulating Inter-player Shrinkage (Player 2 moving towards Player 1)...")
    p2_id = "p2_opponent"
    p2_base = torch.randn(512)
    p2_base = torch.nn.functional.normalize(p2_base, p=2, dim=0)
    
    # Established baseline for P2
    for i in range(10):
        monitor.add_embedding(p2_id, p2_base)
        
    # Start P2 moving towards P1
    p2_moving = p2_base.clone()
    for i in range(10):
        p2_moving = p2_moving * 0.9 + base_emb * 0.1 # Move 10% towards P1 per step
        p2_moving = torch.nn.functional.normalize(p2_moving, p=2, dim=0)
        sig = monitor.add_embedding(p2_id, p2_moving)
        print(f" Frame {i+1}: Inter-Shrinkage: {sig['inter_shrinkage']:.4f}")

    print("\n--- 🏭 Industrial Monitoring Verified ---")

if __name__ == "__main__":
    simulate_industrial_monitoring()
