import numpy as np
import torch
from collections import deque

class ReIDDriftMonitor:
    """
    Industrial-Grade Drift Monitor for Player ReID.
    Tracks embedding centroids over time to detect lighting changes, 
    jersey variance, or identity drift.
    """
    def __init__(self, feature_dim=512, window_size=100, drift_threshold=0.08, ema_alpha=0.01):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.ema_alpha = ema_alpha
        
        # Registry: ID -> Deque of embeddings
        self.history = {}
        # Base Centroids: ID -> First window mean
        self.base_centroids = {}
        # EMA Centroids: ID -> Exponential Moving Average
        self.ema_centroids = {}
        # Global Centroid: Mean of all active players
        self.global_centroid = None
        
    def add_embedding(self, player_id, embedding):
        """
        Adds a new L2-normalized embedding for a player.
        Returns detection signals dict.
        """
        if isinstance(embedding, torch.Tensor):
            norm = torch.norm(embedding).item()
            embedding = embedding.detach().cpu().numpy()
        else:
            norm = np.linalg.norm(embedding)
            
        if player_id not in self.history:
            self.history[player_id] = deque(maxlen=self.window_size)
            
        self.history[player_id].append(embedding)
        
        signals = {
            "intra_drift": 0.0,
            "ema_drift": 0.0,
            "norm_deviation": abs(1.0 - norm),
            "inter_shrinkage": 0.0,
            "alert": False
        }

        # Update EMA
        if player_id not in self.ema_centroids:
            self.ema_centroids[player_id] = embedding
        else:
            self.ema_centroids[player_id] = (1 - self.ema_alpha) * self.ema_centroids[player_id] + self.ema_alpha * embedding
            self.ema_centroids[player_id] /= np.linalg.norm(self.ema_centroids[player_id])

        # If we have a full window, we can check for drift
        if len(self.history[player_id]) == self.window_size:
            current_centroid = np.mean(self.history[player_id], axis=0)
            current_centroid /= np.linalg.norm(current_centroid)
            
            if player_id not in self.base_centroids:
                self.base_centroids[player_id] = current_centroid
            
            # 1. Intra-player drift (vs baseline)
            baseline = self.base_centroids[player_id]
            signals["intra_drift"] = 1.0 - np.dot(current_centroid, baseline)
            
            # 2. EMA drift (short-term shift)
            signals["ema_drift"] = 1.0 - np.dot(current_centroid, self.ema_centroids[player_id])
            
            # 3. Inter-player shrinkage
            # Check if this player's centroid is moving closer to other players
            if len(self.base_centroids) > 1:
                other_pids = [p for p in self.base_centroids if p != player_id]
                current_min_dist = min([1.0 - np.dot(current_centroid, self.base_centroids[p]) for p in other_pids])
                base_min_dist = min([1.0 - np.dot(baseline, self.base_centroids[p]) for p in other_pids])
                signals["inter_shrinkage"] = base_min_dist - current_min_dist

            if signals["intra_drift"] > self.drift_threshold:
                signals["alert"] = True
            
        # Update Global Centroid
        all_centroids = list(self.ema_centroids.values())
        if all_centroids:
            self.global_centroid = np.mean(all_centroids, axis=0)
            self.global_centroid /= np.linalg.norm(self.global_centroid)

        return signals

    def reset_baseline(self, player_id=None):
        """Resets baseline for a specific player or all players."""
        if player_id:
            if player_id in self.base_centroids:
                del self.base_centroids[player_id]
        else:
            self.base_centroids = {}

class IndustrialMetrics:
    """
    Implements High-Level Industrial ReID Metrics:
    - False Merge Rate
    - False Split Rate
    - Fragmentation Index
    """
    @staticmethod
    def compute_fragmentation_index(tracklets_per_id):
        """
        Average number of tracklets assigned to a single ground-truth ID.
        Ideal = 1.0.
        """
        if not tracklets_per_id:
            return 0.0
        return np.mean([len(tracks) for tracks in tracklets_per_id.values()])

    @staticmethod
    def compute_merge_split_rates(pred_ids, gt_ids):
        """
        Simplified Industrial Merge/Split analyzer.
        """
        # Logic to be expanded with Hungarian matching on overlaps
        pass
