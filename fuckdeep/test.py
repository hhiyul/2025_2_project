import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, ClassLabel, DatasetDict
import numpy as np
from datasets import load_from_disk
from torch.amp import autocast, GradScaler

import math
import time
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
from sklearn.metrics import f1_score
from collections import Counter

final_dataset = load_from_disk("C:/Users/USER-PC/Desktop/deep")
labels = [int(x) for x in final_dataset["train"]["label"]]  # numpy나 tensor → int 변환
label_names = final_dataset["train"].features["label"].names
counts = Counter(labels)

print(f"총 클래스 수: {len(label_names)}\n")
for i, name in enumerate(label_names):
    print(f"{i:2d}: {name:20s} → {counts[i]}개")