import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """
    Temporal Attention for tracklet embedding aggregation.
    Weighting frames based on their quality/stability.
    """
    def __init__(self, feat_dim=512):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, tracklet_features):
        """
        Args:
            tracklet_features: (batch, num_frames, feat_dim)
        Returns:
            aggregated_feat: (batch, feat_dim)
        """
        # Compute attention weights
        weights = self.attention(tracklet_features) # (batch, num_frames, 1)
        
        # Weighted sum of features
        aggregated_feat = torch.sum(tracklet_features * weights, dim=1)
        return aggregated_feat

class GlobalIDRegistry:
    """
    Manages ID persistence across multiple camera streams.
    """
    def __init__(self, matching_threshold=0.7):
        self.registry = {} # ID -> cumulative feature
        self.matching_threshold = matching_threshold
        self.next_id = 1

    def match_and_update(self, tracklet_feat):
        """
        Match a new tracklet feature to the global registry.
        """
        best_id = None
        min_dist = float('inf')

        for global_id, stored_feat in self.registry.items():
            dist = torch.norm(tracklet_feat - stored_feat)
            if dist < min_dist and dist < self.matching_threshold:
                min_dist = dist
                best_id = global_id

        if best_id is None:
            best_id = self.next_id
            self.registry[best_id] = tracklet_feat
            self.next_id += 1
        else:
            # Moving average update for the registry feature
            self.registry[best_id] = 0.9 * self.registry[best_id] + 0.1 * tracklet_feat
            
        return best_id
