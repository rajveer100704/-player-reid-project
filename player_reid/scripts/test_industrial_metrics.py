import numpy as np
from collections import defaultdict

def compute_industrial_metrics(gt_identities, pred_identities):
    """
    Computes industrial-grade ReID metrics.
    
    gt_identities: List of ground-truth IDs for each detection.
    pred_identities: List of predicted IDs for each detection.
    
    Returns:
        Fragmentation Index
        Merge Rate
        Split Rate
    """
    # 1. Fragmentation Index
    # How many predicted IDs map to one GT ID?
    gt_to_pred = defaultdict(set)
    for gt, pred in zip(gt_identities, pred_identities):
        gt_to_pred[gt].add(pred)
    
    fragments = [len(preds) for preds in gt_to_pred.values()]
    fragmentation_index = np.mean(fragments)
    
    # 2. Split Rate (normalized fragmentation)
    # % of GT IDs that were split into >1 predicted ID
    splits = sum(1 for f in fragments if f > 1)
    split_rate = splits / len(gt_identities.unique()) if len(gt_identities.unique()) > 0 else 0
    
    # 3. Merge Rate
    # % of predicted IDs that contain >1 GT ID
    pred_to_gt = defaultdict(set)
    for gt, pred in zip(gt_identities, pred_identities):
        pred_to_gt[pred].add(gt)
    
    merges = sum(1 for gts in pred_to_gt.values() if len(gts) > 1)
    merge_rate = merges / len(pred_to_gt) if len(pred_to_gt) > 0 else 0
    
    return fragmentation_index, split_rate, merge_rate

def test_industrial_metrics():
    print("--- 📊 Testing Industrial ReID Metrics ---")
    
    # Simulate a 100-frame track of 3 players
    # GT IDs: 0, 1, 2
    
    # Case A: Perfect Tracking
    print("\nCase A: Perfect Tracking")
    gt = np.array([0, 1, 2] * 100)
    pred = np.array([0, 1, 2] * 100)
    frag, split, merge = compute_industrial_metrics(torch.tensor(gt), torch.tensor(pred))
    print(f" Frag Index: {frag:.2f} (Target 1.0) | Split: {split:.2%} | Merge: {merge:.2%}")
    
    # Case B: Fragmentation (ID Switch)
    # Player 1 (GT 1) switches to ID 10 halfway through
    print("\nCase B: Identity Split (ID Switch)")
    pred_frag = pred.copy()
    pred_frag[150::3] = 10 # Every 3rd detection for GT 1 after index 150
    frag, split, merge = compute_industrial_metrics(torch.tensor(gt), torch.tensor(pred_frag))
    print(f" Frag Index: {frag:.2f} | Split: {split:.2%} | Merge: {merge:.2%}")
    
    # Case C: Identity Merge (Identity Collapse)
    # Player 1 and Player 2 both assigned ID 1
    print("\nCase C: Identity Merge (Collapse)")
    pred_merge = pred.copy()
    pred_merge[gt == 2] = 1 # Map player 2 to player 1's ID
    frag, split, merge = compute_industrial_metrics(torch.tensor(gt), torch.tensor(pred_merge))
    print(f" Frag Index: {frag:.2f} | Split: {split:.2%} | Merge: {merge:.2%}")

if __name__ == "__main__":
    import torch # For .unique() and tensor helper
    test_industrial_metrics()
