from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict

import torch
from torch import nn
from torchvision import transforms


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class DepthwiseConv2d(nn.Module):
    """Depth-wise convolution used inside LPU blocks."""

    def __init__(self, channels: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, k, s, p, groups=channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.dw(x)


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.act(self.bn(self.conv(x)))


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.downsample = ConvBNGELU(in_ch, out_ch, k=3, s=2, p=1)
        self.body = ConvBNGELU(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.downsample(x)
        return self.body(x)


class LPU(nn.Module):
    """Local perception unit used before transformer stages."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = DepthwiseConv2d(channels, k=3, s=1, p=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.dw(x)
        x = self.bn(x)
        return self.act(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 3.0, drop: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 3.0,
            attn_drop: float = 0.0,
            proj_drop: float = 0.1,
            drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)
        self.dp1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dp1(self.drop1(y))
        return x + self.dp2(self.mlp(self.norm2(x)))


class CMTClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            stem_channels: int = 64,
            c_stage1: int = 96,
            c_stage2: int = 128,
            c_stage3: int = 160,
            t_dim1: int = 256,
            t_heads1: int = 4,
            t_depth1: int = 3,
            t_mlp1: float = 3.0,
            t_dim2: int = 384,
            t_heads2: int = 6,
            t_depth2: int = 6,
            t_mlp2: float = 3.5,
            attn_drop: float = 0.0,
            proj_drop: float = 0.1,
            drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            ConvBNGELU(3, stem_channels // 2, k=3, s=2, p=1),
            ConvBNGELU(stem_channels // 2, stem_channels, k=3, s=1, p=1),
        )

        self.stage1 = ConvStage(stem_channels, c_stage1)
        self.stage2 = ConvStage(c_stage1, c_stage2)
        self.stage3 = ConvStage(c_stage2, c_stage3)

        self.to_embed1 = nn.Conv2d(c_stage3, t_dim1, kernel_size=1, stride=1, padding=0, bias=True)

        self.lpu1 = LPU(t_dim1)
        dpr1 = torch.linspace(0, drop_path_rate * 0.5, steps=t_depth1).tolist()
        self.trans1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=t_dim1,
                    num_heads=t_heads1,
                    mlp_ratio=t_mlp1,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr1[i],
                )
                for i in range(t_depth1)
            ]
        )

        self.down_tokens = nn.Sequential(
            nn.Conv2d(t_dim1, t_dim2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(t_dim2, eps=1e-5, momentum=0.1),
            nn.GELU(),
        )

        self.lpu2 = LPU(t_dim2)
        dpr2 = torch.linspace(drop_path_rate * 0.5, drop_path_rate, steps=t_depth2).tolist()
        self.trans2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=t_dim2,
                    num_heads=t_heads2,
                    mlp_ratio=t_mlp2,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr2[i],
                )
                for i in range(t_depth2)
            ]
        )

        self.head_norm = nn.LayerNorm(t_dim2, eps=1e-6)
        self.fc = nn.Linear(t_dim2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.to_embed1(x)

        x = self.lpu1(x)
        bsz, channels, height, width = x.shape  #원본코드랑 변수명 조금씩 다름
        x = x.flatten(2).transpose(1, 2)
        x = self.trans1(x)
        x = x.transpose(1, 2).view(bsz, channels, height, width)

        x = self.down_tokens(x)

        x = self.lpu2(x)
        bsz, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.trans2(x)

        x = x.mean(dim=1)
        x = self.head_norm(x)
        return self.fc(x)


VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint type: expected dict or nn.Module")

    possible_keys = ("model_state_dict", "state_dict", "model")
    for key in possible_keys:
        value = checkpoint.get(key)  # type: ignore[call-arg]
        if isinstance(value, dict):
            return value  # type: ignore[return-value]
    return checkpoint  # type: ignore[return-value]


def load_cmt_model(model_path: Path, num_classes: int, device: torch.device) -> Tuple[nn.Module, Tuple[str, ...], Tuple[str, ...]]:
    """
    - 체크포인트의 다양한 형태(dict/nn.Module) 지원
    - DataParallel 등의 'module.' 접두사 제거
    - stage*.ds.*  <->  stage*.downsample.* 키 자동 리매핑
    - 먼저 strict=True로 로드 → 실패 시 리매핑 시도 → 그래도 안 되면 오류 메시지에 누락/예상치 못한 키를 자세히 포함
    - fc(out_features)와 num_classes 불일치 시 명확한 예외
    """
    ckpt = torch.load(model_path, map_location=device)

    def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
        if isinstance(obj, nn.Module):
            return obj.state_dict()
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
        for key in ("model_state_dict", "state_dict", "model"):
            v = obj.get(key)
            if isinstance(v, dict):
                return v
        return obj  # raw state_dict

    def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 'module.'로 시작하면 제거
        if any(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd

    def _remap_ds_downsample(sd: Dict[str, torch.Tensor], to: str) -> Dict[str, torch.Tensor]:
        """
        to == 'downsample' : stage*.ds.* -> stage*.downsample.*
        to == 'ds'         : stage*.downsample.* -> stage*.ds.*
        """
        out = {}
        for k, v in sd.items():
            if to == "downsample":
                k2 = (k.replace("stage1.ds.", "stage1.downsample.")
                      .replace("stage2.ds.", "stage2.downsample.")
                      .replace("stage3.ds.", "stage3.downsample."))
            else:
                k2 = (k.replace("stage1.downsample.", "stage1.ds.")
                      .replace("stage2.downsample.", "stage2.ds.")
                      .replace("stage3.downsample.", "stage3.ds."))
            out[k2] = v
        return out

    # 1) state_dict 정규화
    sd = _extract_state_dict(ckpt)
    sd = _strip_module_prefix(sd)

    # 2) 모델 생성
    model = CMTClassifier(num_classes=num_classes)

    # 3) fc(out_features)와 num_classes 일치 검증 (실패 시 바로 알려줌)
    fc_out = getattr(model, "fc", None)
    if hasattr(fc_out, "out_features") and fc_out.out_features != num_classes:
        raise ValueError(
            f"[load_cmt_model] num_classes({num_classes}) != model.fc.out_features({fc_out.out_features}). "
            "훈련 당시 라벨 개수/순서와 현재 labels.json이 다른지 확인하세요."
        )

    # 4) 우선 strict=True로 그대로 시도
    try:
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
        return model, (), ()
    except RuntimeError as e1:
        # 5) 키 패턴 보고 리매핑 방향 결정
        keys = list(sd.keys())
        has_ds = any(".ds." in k for k in keys)
        has_down = any(".downsample." in k for k in keys)

        # 현재 코드가 downsample를 쓰는 경우가 대부분이므로, 체크포인트가 ds면 downsample로 리매핑
        if has_ds and not has_down:
            sd2 = _remap_ds_downsample(sd, to="downsample")
        elif has_down and not has_ds:
            # 혹시 반대 상황이면 반대로
            sd2 = _remap_ds_downsample(sd, to="ds")
        else:
            sd2 = sd  # 패턴 판별 불가 → 원본 유지

        try:
            model.load_state_dict(sd2, strict=True)
            model.to(device).eval()
            return model, (), ()
        except RuntimeError as e2:
            # 6) 디버그용으로 missing/unexpected 키를 뽑아주기 위해 strict=False로 한 번 계산
            missing, unexpected = model.load_state_dict(sd2, strict=False)
            # 더 명확한 에러 메시지로 실패 이유 전달
            raise RuntimeError(
                "[load_cmt_model] state_dict 로딩 실패\n"
                f"- 1차(strict=True, 원본) 오류: {e1}\n"
                f"- 2차(strict=True, 리매핑) 오류: {e2}\n"
                f"- 참고: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

__all__ = [
    "CMTClassifier",
    "DropPath",
    "VAL_TRANSFORM",
    "load_cmt_model",
]