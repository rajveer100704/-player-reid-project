import torch
import torch.nn as nn
from player_reid.models.arcface_loss import ArcFaceLoss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining."""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            if mask[i].any():
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if (mask[i] == 0).any():
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        
        if not dist_ap or not dist_an:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class HybridLoss(nn.Module):
    """Combined ArcFace and Triplet Loss."""
    def __init__(self, num_classes, feat_dim=512, margin=0.3, s=30.0, m=0.5):
        super(HybridLoss, self).__init__()
        self.arcface = ArcFaceLoss(num_classes, feat_dim, s, m)
        self.triplet = TripletLoss(margin)

    def forward(self, bn_feat, feat, targets):
        loss_arc = self.arcface(bn_feat, targets)
        loss_tri = self.triplet(feat, targets)
        return loss_arc + loss_tri
