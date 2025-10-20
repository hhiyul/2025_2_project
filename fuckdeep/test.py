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
from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset():
    dataset = load_dataset("Densu341/Fresh-rotten-fruit")


    remove_labels = [18, 20, 16, 13, 2, 5, 7, 9]
    labels = np.array(dataset["train"]["label"])
    mask = ~np.isin(labels, remove_labels)

    # 3. 필요 없는 라벨 제거
    clean_dataset = dataset["train"].select(np.where(mask)[0])

    # 4. train/val split
    dataset = clean_dataset.train_test_split(test_size=0.2)
    train_dataset, val_dataset = dataset["train"], dataset["test"]

    # 5. 실제 남은 라벨 인덱스 및 이름 추출
    unique_labels = sorted(set(train_dataset["label"]) | set(val_dataset["label"]))
    all_labels = [train_dataset.features["label"].int2str(i) for i in unique_labels]

    # 6. 새로운 ClassLabel 정의
    new_classlabel = ClassLabel(num_classes=len(all_labels), names=all_labels)

    # 7. 라벨 값 재매핑
    def remap_labels(example):
        label_name = train_dataset.features["label"].int2str(example["label"])
        example["label"] = all_labels.index(label_name)
        return example

    train_dataset = train_dataset.map(remap_labels)
    val_dataset   = val_dataset.map(remap_labels)

    train_dataset = train_dataset.cast_column("label", new_classlabel)
    val_dataset   = val_dataset.cast_column("label", new_classlabel)

    # 8. 최종 DatasetDict 생성
    final_dataset = DatasetDict({
        "train": train_dataset,
        "test": val_dataset
    })
    return final_dataset


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])  # ImageNet 기준
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# --------------------------------------------------
# PyTorch Dataset 래퍼
# --------------------------------------------------
class FruitHFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")  # RGBA → RGB
        label = sample["label"]                # 이미 0~N-1 정수
        if self.transform:
            image = self.transform(image)
        return image, label


# ------------------------------------------------------------
# 3)  CNN (BatchNorm/Dropout/정규화 없음) ㅈㄴ간단
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),   # 112x112
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),  # 56x56
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), # 28x28
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x  # raw logits
    

# ------------------------------------------------------------
# 4) 3-Fold 분할 & 학습 루프 (bs=32, epochs=5)
# ------------------------------------------------------------
def main():
    final_dataset = prepare_dataset()
    EPOCHS = 5
    BATCH_SIZE = 32  #데스크탑은 128로
    K = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

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

        train_ds = FruitHFDataset(train_split, transform=train_transform)
        val_ds   = FruitHFDataset(val_split,   transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=(4>0), prefetch_factor=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=(4>0), prefetch_factor=2)
        
        torch.backends.cudnn.benchmark = True

        # --- 모델/손실/옵티마이저 ---
        model = SmallCNN(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scaler = GradScaler()

        val_acc_list = []

        # --- Epoch 루프 ---
        for epoch in range(1, EPOCHS+1):
            epoch_start = time.time()
            print(f"\n▶ Fold {fold}/{K} | Epoch {epoch}/{EPOCHS}")

            # ---- [Train] ----
            model.train()
            total, correct, loss_sum = 0, 0, 0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} [Train]", ncols=100)

            for x, y in pbar:
                # GPU 전송 최적화
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda"): 
                    out = model(x)
                    loss = criterion(out, y)

                # ✅ AMP: scaled backward & step
                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()

                    # 통계
                    bs = x.size(0)
                    loss_sum += loss.item() * bs
                    correct  += (out.argmax(1) == y).sum().item()
                    total    += bs

            tr_acc  = correct / max(1, total)
            tr_loss = loss_sum / max(1, total)
            print(f"Train ▶ acc: {tr_acc:.4f} | loss: {tr_loss:.4f}")
                # ---- [Validation] ----
            model.eval()
            v_total = v_correct = 0
            v_loss_sum = 0.0
            with torch.no_grad():
                with autocast("cuda"):
                    for x, y in tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch} [Val]", ncols=100):
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        out = model(x)
                        loss = criterion(out, y)

                        v_loss_sum += loss.item() * x.size(0)
                        v_correct  += (out.argmax(1) == y).sum().item()
                        v_total    += x.size(0)

                    va_acc  = v_correct / max(1, v_total)
                    va_loss = v_loss_sum / max(1, v_total)
                    val_acc_list.append(va_acc)
                    print(f"Val ▶ acc: {va_acc:.4f} | loss: {va_loss:.4f}")

            epoch_time = time.time() - epoch_start
            print(f"⏱️  Epoch {epoch} 완료 (소요시간: {epoch_time:.2f}초)")

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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
