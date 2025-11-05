import os, json, glob
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from test_copy import CMTClassifier, prepare_dataset   # prepare_dataset은 ②방법 쓸 때만 필요

CKPT_PATH   = "C:/Users/rkdrn/Desktop/deep/best_model_fold1.pt"
TEST_IMAGE  = "C:/Users/rkdrn/Desktop/tset/2.jpg"  # 테스트용 이미지
LABEL_JSON  = "C:/Users/rkdrn/Desktop/deep/label_names.json" 

with open(LABEL_JSON, "r", encoding="utf-8") as f:
    label_names = json.load(f)
TEST_DIR    = None  # r"C:\path\to\test_images"  # 폴더 단위 추론하고 싶으면 경로 넣기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
# ===== 유틸 =====
def safe_torch_load(path, map_location):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(f"가중치 파일 이상: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def get_label_names():
    # ⚠️ 테스트에서는 prepare_dataset를 절대 호출하지 않음
    if LABEL_JSON and os.path.isfile(LABEL_JSON):
        with open(LABEL_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None  # 없으면 인덱스로 출력

def infer_num_classes_from_state_dict(state):
    for k, v in state.items():
        if k.endswith("fc.weight") and v.ndim == 2:
            return v.shape[0]
    raise RuntimeError("fc.weight에서 클래스 수 추정 실패")

def predict_image(model, TEST_IMAGE, label_names):
    if not os.path.isfile(TEST_IMAGE):
        raise FileNotFoundError(TEST_IMAGE)
    img = Image.open(TEST_IMAGE).convert("RGB")
    x = VAL_TF(img).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        probs = F.softmax(model(x), dim=1).squeeze(0)  # (C,)
        conf, idx = probs.max(dim=0)
    idx = int(idx.item()); conf = float(conf.item())
    label = label_names[idx] if (label_names is not None and 0 <= idx < len(label_names)) else str(idx)
    return label, conf

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
def list_images(root):
    paths = [p for p in glob.glob(os.path.join(root, "*")) if os.path.splitext(p.lower())[1] in IMG_EXTS]
    paths.sort()
    return paths

def predict_folder(model, folder_path: str, label_names, batch_size=128, num_workers=0, pin_memory=True):
    # num_workers=0 권장(테스트 스크립트 안정성)
    class FolderDS(Dataset):
        def __init__(self, paths, tf):
            self.paths = paths; self.tf = tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]
            x = VAL_TF(Image.open(p).convert("RGB"))
            return x, p

    paths = list_images(folder_path)
    ds = FolderDS(paths, VAL_TF)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    results = []
    with torch.inference_mode():
        for x, ps in loader:
            x = x.to(device, non_blocking=True)
            probs = F.softmax(model(x), dim=1)
            conf, idx = probs.max(dim=1)
            for pth, i, c in zip(ps, idx.cpu().tolist(), conf.cpu().tolist()):
                name = label_names[i] if (label_names is not None and 0 <= i < model.fc.out_features) else str(i)
                results.append((pth, name, c))
    return results

if __name__ == "__main__":  # ✅ Windows 멀티프로세싱 재실행 방지
    print("device:", device)

    # 1️⃣ 가중치 로드
    state = safe_torch_load(CKPT_PATH, device)
    label_names = get_label_names()
    
    num_classes = len(label_names)

    # 2️⃣ 모델 준비
    model = CMTClassifier(num_classes=num_classes).to(device).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[경고] state_dict 키 불일치:", "missing:", missing, "unexpected:", unexpected)

    # 3️⃣ 단일 이미지 추론  ✅ (model, TEST_IMAGE, label_names 모두 넘겨야 함)
    if TEST_IMAGE:
        lab, pr = predict_image(model, TEST_IMAGE, label_names)
        print(f"[단일] {TEST_IMAGE} → {lab} ({pr*100:.2f}%)")
"""
# ===== 실행 예시 =====
print("device:", device)
if TEST_IMAGE:
    lab, pr = predict_image(model, TEST_IMAGE, label_names)
    print(f"[단일] {TEST_IMAGE} → {lab} ({pr*100:.2f}%)")
    """