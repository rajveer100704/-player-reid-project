import torch
import torch.nn as nn
from torch.nn import functional as F
from player_reid.models.bnneck import BNNeck

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, relu=True, bn=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.relu: x = self.relu(x)
        return x

class OSBlock(nn.Module):
    """Omni-Scale Residual Block"""
    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = ConvLayer(in_channels, mid_channels, 1)
        
        self.stream1 = nn.Sequential(ConvLayer(mid_channels, mid_channels, 3, padding=1))
        self.stream2 = nn.Sequential(
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1)
        )
        self.stream3 = nn.Sequential(
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1)
        )
        self.stream4 = nn.Sequential(
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1),
            ConvLayer(mid_channels, mid_channels, 3, padding=1)
        )
        
        self.gate = ConvLayer(mid_channels, mid_channels, 1)
        self.conv2 = ConvLayer(mid_channels, out_channels, 1, relu=False)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, 1, relu=False)

    def forward(self, x):
        identity = self.shortcut(x)
        x1 = self.conv1(x)
        s1 = self.stream1(x1)
        s2 = self.stream2(x1)
        s3 = self.stream3(x1)
        s4 = self.stream4(x1)
        x = s1 + s2 + s3 + s4
        x = self.gate(x)
        x = self.conv2(x)
        return F.relu(x + identity)

class OSNet(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(OSNet, self).__init__()
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = OSBlock(64, 64)
        self.layer2 = OSBlock(64, 128)
        self.layer3 = OSBlock(128, 256)
        self.layer4 = OSBlock(256, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # BNNeck Design (The 10/10 Secret Sauce)
        self.bnneck = BNNeck(feature_dim)
        
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        
        # Features for Triplet Loss (before BN)
        feat = v
        # Normalized features for Inference/Metric matching
        bn_feat = self.bnneck(v)
        
        if not self.training:
            return bn_feat
        
        return bn_feat, feat

def create_osnet(num_classes=751):
    return OSNet(num_classes)
