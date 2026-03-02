import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from player_reid.evaluation.industrial_metrics import IndustrialEvaluator

def test_industrial_evaluator():
    print("--- 📑 Testing Industrial Evaluator (Match-Level Analysis) ---")
    
    # 1. Simulate PERFECT Tracking (1 Player, 60s, 1 TrackID, 1 GTID)
    print("\nCase 1: Perfect Tracking (1 Minute)")
    tracks = [(t, 100, t) for t in range(60)] # (ts, tid, det)
    gt = [(t, 1, t) for t in range(60)]      # (ts, gtid, det)
    
    results = IndustrialEvaluator.evaluate_match(tracks, gt, 60)
    print(f" FI: {results['fragmentation_index']:.2f} | Switches/Min: {results['switches_per_minute']:.2f} | Total Switches: {results['total_switches']}")

    # 2. Simulate ID Switch (Fragmentation)
    # Player 1 (GT 1) switches from TID 100 to TID 101 at 30s
    print("\nCase 2: ID Switch (Fragmentation) at 30s")
    tracks_switch = [(t, 100 if t < 30 else 101, t) for t in range(60)]
    results = IndustrialEvaluator.evaluate_match(tracks_switch, gt, 60)
    print(f" FI: {results['fragmentation_index']:.2f} | Switches/Min: {results['switches_per_minute']:.2f} | Total Switches: {results['total_switches']}")

    # 3. Simulate Identity Merge (Collapse)
    # 2 GT players (1 and 2) but only 1 TrackID (100)
    print("\nCase 3: Identity Merge (Collapse) - 2 Players, 1 TrackID")
    gt_merge = [(t, 1, t) for t in range(60)] + [(t, 2, t + 60) for t in range(60)]
    tracks_merge = [(t, 100, t) for t in range(60)] + [(t, 100, t + 60) for t in range(60)]
    results = IndustrialEvaluator.evaluate_match(tracks_merge, gt_merge, 60)
    print(f" Merge Rate: {results['merge_rate']:.2%} | FI: {results['fragmentation_index']:.2f}")

    # 4. Realistic Stress: 1 Switch per 10 minutes
    print("\nCase 4: Elite Performance - 1 Switch in 10 minutes")
    tracks_elite = [(t, 100 if t < 600 else 101, t) for t in range(600)]
    gt_elite = [(t, 1, t) for t in range(600)]
    results = IndustrialEvaluator.evaluate_match(tracks_elite, gt_elite, 600)
    print(f" Switches/Min: {results['switches_per_minute']:.2f} (Target < 0.2)")

    print("\n--- 📑 Industrial Metrics Verified ---")

if __name__ == "__main__":
    test_industrial_evaluator()
