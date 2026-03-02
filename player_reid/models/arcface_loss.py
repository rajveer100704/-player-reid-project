import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)
    Reference: Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
    """
    def __init__(self, num_classes, feat_dim=512, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix (L2 normalized) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, targets)
