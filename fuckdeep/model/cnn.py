import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class SeparableConv(nn.Module):
    def __init__(self, cin, cout, stride=1, drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, kernel_size=3, stride=stride, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.dw(x); x = self.dw_bn(x); x = self.act(x)
        x = self.pw(x); x = self.pw_bn(x); x = self.act(x)
        x = self.dropout(x)
        return x

class SmallCNN(nn.Module):
    """
    입력: (B,3,224,224) 가정
    다운샘플: /2 → /2 → /2 → /2 → /2 (총 /32, 224→7)
    채널: 32 → 64 → 128 → 192 → 256 (가벼움)
    분류: GAP → Dropout → Linear
    """
    def __init__(self,
                 num_classes: Optional[int] = None,
                 multitask: bool = False,
                 num_fruit: int = 7,
                 num_fresh: int = 2,
                 drop_block: float = 0.05,
                 drop_head: float = 0.2):
        super().__init__()
        self.multitask = multitask

        chs = [32, 64, 128, 192, 256]  # 가벼운 채널 구성
        self.stem = nn.Sequential(
            nn.Conv2d(3, chs[0], kernel_size=3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = SeparableConv(chs[0], chs[1], stride=2, drop=drop_block)     # 112 -> 56
        self.stage2 = SeparableConv(chs[1], chs[2], stride=2, drop=drop_block)     # 56  -> 28
        self.stage3 = SeparableConv(chs[2], chs[3], stride=2, drop=drop_block)     # 28  -> 14
        self.stage4 = SeparableConv(chs[3], chs[4], stride=2, drop=drop_block)     # 14  -> 7

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(p=drop_head)

        feat_dim = chs[-1]
        if multitask:
            self.head_fruit = nn.Linear(feat_dim, num_fruit)
            self.head_fresh = nn.Linear(feat_dim, num_fresh)
        else:
            assert num_classes is not None, "num_classes 를 지정하세요 (단일 과제 모드)."
            self.head = nn.Linear(feat_dim, num_classes)

        # 간단 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)              # (B, C, 7, 7)
        x = self.gap(x).squeeze(-1).squeeze(-2)  # (B, C)
        x = torch.flatten(x, 1)
        x = self.head_drop(x)

        if self.multitask:
            return self.head_fruit(x), self.head_fresh(x)
        else:
            return self.head(x), None