import torch
import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class BNNeck(nn.Module):
    """
    BNNeck Design: Decouples the feature space and the classification space.
    Reference: Luo et al. Bag of Tricks and a Strong Baseline for Deep Person Re-Identification. CVPR 2019.
    """
    def __init__(self, feature_dim=512):
        super(BNNeck, self).__init__()
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck(x)
