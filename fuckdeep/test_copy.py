import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, ClassLabel, DatasetDict
import numpy as np
from torch.amp import autocast, GradScaler
import math
import time
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score, top_k_accuracy_score
from datasets import load_from_disk
from torch.autograd import Variable
from collections import Counter

import json, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


def prepare_dataset():
    dataset = load_dataset("Densu341/Fresh-rotten-fruit")

    # 1) 라벨 제거
    remove_labels = [18, 20, 16, 13, 2, 5, 7, 9]
    labels = np.array(dataset["train"]["label"])
    mask = ~np.isin(labels, remove_labels)
    clean = dataset["train"].select(np.where(mask)[0])

    # 2) split (결정적)
    split = clean.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    # 3) 라벨 재매핑 (이름 기준으로 0..C-1)
    uniq = sorted(set(train_ds["label"]) | set(val_ds["label"]))
    names = [train_ds.features["label"].int2str(i) for i in uniq]
    new_lbl = ClassLabel(num_classes=len(names), names=names)

    def remap(example):
        name = train_ds.features["label"].int2str(example["label"])
        example["label"] = names.index(name)
        return example

    train_ds = train_ds.map(remap, num_proc=os.cpu_count()//2,
                            load_from_cache_file=True, desc="Remap train")
    val_ds   = val_ds.map(remap,   num_proc=os.cpu_count()//2,
                            load_from_cache_file=True, desc="Remap val")

    train_ds = train_ds.cast_column("label", new_lbl)
    val_ds   = val_ds.cast_column("label",  new_lbl)

    # 4) RGB 통일 (결정적)
    def to_rgb(example):
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        example["image"] = img
        return example

    train_ds = train_ds.map(to_rgb, num_proc=os.cpu_count()//2,
                            load_from_cache_file=True, desc="RGB train")
    val_ds   = val_ds.map(to_rgb,   num_proc=os.cpu_count()//2,
                            load_from_cache_file=True, desc="RGB val")

    # ✅ 여기서 끝! set_transform 안 씀
    return DatasetDict({"train": train_ds, "test": val_ds})


# PyTorch Dataset 래퍼
# --------------------------------------------------
class FruitHFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.tf = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]            # image: PIL.Image, label: int
        img  = item["image"]
        if self.tf is not None:
            img = self.tf(img)         # Tensor(C,H,W)
        label = item["label"]
        # long 보장
        import torch
        if not torch.is_tensor(label):
            import torch
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.to(dtype=torch.long)
        return img, label


# ------------------------------------------------------------
# 모델 코드임 cmt기반 cnn+t      ㅈㄴ복잡
# ------------------------------------------------------------
class DropPath(nn.Module):
    """ per-sample DropPath (Stochastic Depth) """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # x: (B, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor

class DepthwiseConv2d(nn.Module):
    def __init__(self, channels, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, k, s, p, groups=channels, bias=bias)

    def forward(self, x):
        return self.dw(x)

class ConvBNGELU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ds   = ConvBNGELU(in_ch, out_ch, k=3, s=2, p=1)   # downsample
        self.body = ConvBNGELU(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        x = self.ds(x)
        x = self.body(x)
        return x
    
class LPU(nn.Module):
    """ Local Perception Unit: 3x3 depthwise → GELU → BN (채널 보존) """
    def __init__(self, channels):
        super().__init__()
        self.dw  = DepthwiseConv2d(channels, k=3, s=1, p=1, bias=False)
        self.bn  = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, drop=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3.0, attn_drop=0.0, proj_drop=0.1, drop_path=0.0): #drop_path는 에포크 늘어나면 늘리삼
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) #1e-5 ~ 1e-7 사이 값
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)
        self.dp1   = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.dp2   = DropPath(drop_path)

    def forward(self, x):
        # x: (B, N, C)
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dp1(self.drop1(y))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x

    

class CMTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        # CNN channels
        stem_channels: int = 64,
        c_stage1: int = 96,
        c_stage2: int = 128,
        c_stage3: int = 160,
        # Transformer dims/heads/depths
        t_dim1: int = 256,  t_heads1: int = 4,  t_depth1: int = 3,  t_mlp1: float = 3.0,
        t_dim2: int = 384,  t_heads2: int = 6,  t_depth2: int = 6,  t_mlp2: float = 3.5,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.0,  # 에폭 짧으면 0.0, 길게 학습하면 0.05~0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # ----- CNN stem (224 -> 112) -----
        self.stem = nn.Sequential(
            ConvBNGELU(3, stem_channels // 2, k=3, s=2, p=1),
            ConvBNGELU(stem_channels // 2, stem_channels, k=3, s=1, p=1),
        )

        # ----- CNN stages (112 -> 56 -> 28 -> 14) -----
        self.stage1 = ConvStage(stem_channels, c_stage1)
        self.stage2 = ConvStage(c_stage1, c_stage2)
        self.stage3 = ConvStage(c_stage2, c_stage3)

        # ----- to embed (14x14, C3 -> D1) -----
        self.to_embed1 = nn.Conv2d(c_stage3, t_dim1, kernel_size=1, stride=1, padding=0, bias=True)

        # ----- Stage A @14x14 : LPU → Transformer(depth=t_depth1) -----
        self.lpu1 = LPU(t_dim1)
        dpr1 = torch.linspace(0, drop_path_rate * 0.5, steps=t_depth1).tolist()
        self.trans1 = nn.Sequential(*[
            TransformerBlock(
                dim=t_dim1, num_heads=t_heads1, mlp_ratio=t_mlp1,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr1[i]
            ) for i in range(t_depth1)
        ])

        # ----- down tokens: 14->7, D1->D2 -----
        self.down_tokens = nn.Sequential(
            nn.Conv2d(t_dim1, t_dim2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(t_dim2, eps=1e-5, momentum=0.1),
            nn.GELU(),
        )

        # ----- Stage B @7x7 : LPU → Transformer(depth=t_depth2) -----
        self.lpu2 = LPU(t_dim2)
        dpr2 = torch.linspace(drop_path_rate * 0.5, drop_path_rate, steps=t_depth2).tolist()
        self.trans2 = nn.Sequential(*[
            TransformerBlock(
                dim=t_dim2, num_heads=t_heads2, mlp_ratio=t_mlp2,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr2[i]
            ) for i in range(t_depth2)
        ])

        # ----- Head -----
        self.head_norm = nn.LayerNorm(t_dim2, eps=1e-6)
        self.fc        = nn.Linear(t_dim2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, "weight") and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:     nn.init.zeros_(m.bias)

    def forward(self, x):
        # CNN 얕은 특징
        x = self.stem(x)          # 224 -> 112
        x = self.stage1(x)        # 112 -> 56
        x = self.stage2(x)        # 56  -> 28
        x = self.stage3(x)        # 28  -> 14

        # 임베딩
        x = self.to_embed1(x)     # B x D1 x 14 x 14

        # Stage A: LPU → Transformer @14x14
        x = self.lpu1(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # B x 196 x D1
        x = self.trans1(x)
        x = x.transpose(1, 2).view(B, C, H, W)     # B x D1 x 14 x 14

        # Downsample tokens: 14 -> 7
        x = self.down_tokens(x)                    # B x D2 x 7 x 7

        # Stage B: LPU → Transformer @7x7
        x = self.lpu2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # B x 49 x D2
        x = self.trans2(x)                         # B x 49 x D2

        # Head
        x = x.mean(dim=1)                          # token 평균 (GAP)
        x = self.head_norm(x)
        logits = self.fc(x)
        return logits
# ------------------------------------------------------------      
# 손실함수 클래스


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class alpha.
    - inputs: logits (B, C)
    - targets: int labels (B,)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", eps=1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        # logits -> log-prob/prob
        log_probs = F.log_softmax(inputs, dim=1)   # (B, C)
        probs     = log_probs.exp()                # (B, C)

        targets = targets.long()
        log_pt  = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt      = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # --- 수치 안정화 ---
        pt = pt.clamp(min=self.eps, max=1. - self.eps)
        # log_pt는 log_softmax 결과라 이미 안정적이지만, 혹시 모를 NaN 방지:
        log_pt = torch.log(pt)

        # alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
        else:
            alpha_t = torch.ones_like(pt)

        # focal term
        focal = (1.0 - pt).pow(self.gamma)
        loss  = -alpha_t * focal * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
# ------------------------------------------------------------
# 메인
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    final_dataset = prepare_dataset()

    names = final_dataset["train"].features["label"].names
    save_dir = "C:/Users/rkdrn/Desktop/deep"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)

        
    num_classes = len(final_dataset["train"].features["label"].names)

    # ----- class-balanced alpha (FocalLoss용) -----
    train_labels = [int(x) for x in final_dataset["train"]["label"]]
    counts = Counter(train_labels)
    class_counts = [counts[i] for i in range(num_classes)]

    beta = 0.999
    effective_num = [1.0 - (beta ** c) for c in class_counts]
    raw_alpha = torch.tensor([(1.0 - beta) / (en if en > 0 else 1e-8)
                              for en in effective_num], dtype=torch.float32)
    alpha = (raw_alpha / raw_alpha.sum()) * num_classes
    print("alpha:", alpha.tolist())


    EPOCHS = 5
    BATCH_SIZE = 128  #데스크탑은 128로
    K = 3
    best_acc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    print(torch.cuda.get_device_name(0))

# StratifiedKFold를 위해 라벨 벡터 추출
    labels  = np.asarray(final_dataset["train"]["label"], dtype=np.int64)
    indices = np.arange(len(final_dataset["train"]), dtype=np.int64)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    fold_accs = []
    start_time = time.time()
    num_classes = len(final_dataset["train"].features["label"].names)

    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), 1):
        print(f"\n================ Fold {fold}/{K} 시작 ================")
        fold_start = time.time()

        # --- (1) 데이터 분할 & DataLoader ---
        train_split = final_dataset["train"].select(list(train_idx))
        val_split   = final_dataset["train"].select(list(val_idx))

        train_ds = FruitHFDataset(final_dataset["train"], transform=train_transform)
        val_ds   = FruitHFDataset(final_dataset["test"],  transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=6, pin_memory=True, persistent_workers=False, prefetch_factor=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=6, pin_memory=True, persistent_workers=False, prefetch_factor=2)
        
        #병목인지 확인하는 코드임---------------------
        loader_start = time.time()
        for i, (x, y) in enumerate(train_loader):
            if i == 10:
                break
        print(f"첫 10 batch 로딩 시간: {time.time() - loader_start:.2f}초")
        #--------------------------------------------

        torch.backends.cudnn.benchmark = True

        # --- 모델/손실/옵티마이저 -----------------------------------------------------------------------
        model = CMTClassifier(num_classes).to(device)
        criterion = FocalLoss(alpha=alpha.to(device), gamma=2.0).to(device) #손실함수로 focal loss 사용함
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) #옵티마이저는 AdamW 사용

        scaler = GradScaler()

                
        val_acc_list = []
        val_loss_list = []
        val_f1_list   = []

        # --- Epoch 루프 ---
        for epoch in range(1, EPOCHS+1):
            epoch_start = time.time()
            print(f"\n▶ Fold {fold}/{K} | Epoch {epoch}/{EPOCHS}")

            # ---- [Train] ----
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} [Train]", ncols=100)

            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda"):
                    out  = model(x)
                    loss = criterion(out, y)

                # (선택) gradient clipping을 원한다면: scaler.unscale_ 후 clip
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = x.size(0)
                loss_sum += loss.item() * bs
                correct  += (out.argmax(1) == y).sum().item()
                total    += bs

            tr_acc  = correct / max(1, total)
            tr_loss = loss_sum / max(1, total)
            print(f"Train ▶ acc: {tr_acc:.4f} | loss: {tr_loss:.4f}")
                # ---- [Validation] ----
            model.eval()
            v_total, v_correct, v_loss_sum = 0, 0, 0.0
            

            all_preds, all_labels, all_logits = [], [], []
            with torch.inference_mode():
                 for x, y in tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch} [Val]", ncols=100):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with autocast("cuda"):
                        out  = model(x)
                        v_loss = criterion(out, y)

                    bs = x.size(0)
                    v_loss_sum += v_loss.item() * bs
                    preds = out.argmax(1)
                    v_correct += (preds == y).sum().item()
                    v_total   += bs

                    # 🔹 예측/정답/로짓 저장 (CPU로 변환)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(y.detach().cpu().numpy())
                    all_logits.append(out.detach().cpu().numpy())

            va_acc  = v_correct / max(1, v_total)
            va_loss = v_loss_sum / max(1, v_total)
            all_logits = np.concatenate(all_logits, axis=0)
            va_f1   = f1_score(all_labels, all_preds, average="macro")
            va_bal  = balanced_accuracy_score(all_labels, all_preds)

            try:
                va_top2 = top_k_accuracy_score(all_labels, all_logits, k=2, labels=np.arange(all_logits.shape[1]))
                va_top3 = top_k_accuracy_score(all_labels, all_logits, k=3, labels=np.arange(all_logits.shape[1]))
            except Exception:
                va_top2 = va_top3 = None

    
            val_acc_list.append(va_acc)
            val_loss_list.append(va_loss)
            val_f1_list.append(va_f1)
            print(f"Val ▶ acc: {va_acc:.4f} | f1: {va_f1:.4f} | bal_acc: {va_bal:.4f} | loss: {va_loss:.4f}")
            if va_top2 is not None:
                print(f"      top-2: {va_top2:.4f} | top-3: {va_top3:.4f}")

            
            save_dir = "C:/Users/rkdrn/Desktop/deep"
            os.makedirs(save_dir, exist_ok=True)
            
            if va_acc > best_acc + 1e-6: 
                best_acc = va_acc
                save_path = os.path.join(save_dir, f"best_model_fold{fold}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved! (fold={fold}, acc={best_acc:.4f})")


            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} 완료 (소요시간: {epoch_time:.2f}초)")

        # --- (4) Fold 종료 처리 (fold ‘안’) ---
        fold_time = time.time() - fold_start
        print(f"✅ Fold {fold} 완료! (소요시간: {fold_time/60:.2f}분)")
        fold_accs.append(val_acc_list)

    # --- (5) 전체 요약 (fold ‘바깥’) ---
    total_time = time.time() - start_time
    print(f"\n================ 학습종료 총 소요시간: {total_time/60:.2f}분 ================")

    print("\n===== K-Fold 결과 요약 =====")
    best_accs = [max(a) if isinstance(a, list) else a for a in fold_accs]
    for i, acc in enumerate(best_accs, start=1):
        print(f"Fold {i:>2} | 최고 val_acc = {acc:.4f}")
    mean_acc = sum(best_accs) / len(best_accs)
    print(f"평균 val_acc = {mean_acc:.4f}")

    torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
