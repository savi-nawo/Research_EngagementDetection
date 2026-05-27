import torch
import torch.nn as nn

def ConvBlock(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True)
    )

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2)   # 416 → 208
        )

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2)   # 208 → 104
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2)   # 104 → 52
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3

class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1, f2, f3):
        return f1, f2, f3   # small, medium, large

class DetectionHead(nn.Module):
    def __init__(self, anchors, num_classes=1):
        super().__init__()

        self.anchors = anchors
        self.num_outputs = 5 + num_classes   # tx, ty, tw, th, obj, cls

        self.small_head  = nn.Conv2d(32,  3*self.num_outputs, 1)
        self.medium_head = nn.Conv2d(64,  3*self.num_outputs, 1)
        self.large_head  = nn.Conv2d(128, 3*self.num_outputs, 1)

    def forward(self, s, m, l):

        def reshape(x):
            B, C, H, W = x.shape
            return x.view(B, 3, self.num_outputs, H, W)

        return [
            reshape(self.small_head(s)),    # 208×208
            reshape(self.medium_head(m)),   # 104×104
            reshape(self.large_head(l))     # 52×52
        ]

class MallDetector(nn.Module):
    def __init__(self, anchors, num_classes=1):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.fusion = MultiScaleFusion()
        self.head = DetectionHead(anchors, num_classes)

    def forward(self, x):
        f1, f2, f3 = self.extractor(x)
        s, m, l = self.fusion(f1, f2, f3)
        return self.head(s, m, l)
