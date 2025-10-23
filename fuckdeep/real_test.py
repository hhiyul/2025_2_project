from test_copy import CMTClassifier
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from datasets import load_from_disk
from collections import OrderedDict

import numpy as np
# 경로/디바이스 설정
# ======================
DATASET_DIR = "C:/Users/USER-PC/Desktop/deep"  # save_to_disk 폴더
CKPT_PATH   = "C:/Users/USER-PC/Desktop/deep/model_data/best_model_fold2.pt"
TEST_IMAGE  = "C:/Users/USER-PC/Pictures/Saved Pictures/v.jpg"  # 테스트용 이미지

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ======================
# 1) 라벨 정의 로드 (훈련 시 사용했던 데이터셋 그대로)
# ======================
final_dataset = load_from_disk(DATASET_DIR)
label_names = final_dataset["train"].features["label"].names
num_classes = len(label_names)
print(f"num_classes (from dataset) = {num_classes}")

# ======================
# 2) 체크포인트 안전 로드 (weights_only 우선, 실패 시 fallback)
# ======================
def load_state_dict_safely(path, map_location):
    # 새 PyTorch(지원) → weights_only=True
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 구버전 PyTorch → weights_only 인자 없음
        sd = torch.load(path, map_location=map_location)
        return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd

state_dict = load_state_dict_safely(CKPT_PATH, device)

# ======================
# 3) (선택) ckpt의 최종 분류기 out_features 확인 → dataset 라벨 수와 일치 점검
# ======================
def detect_out_features(sd):
    # 명시적 'fc.weight'가 있으면 그걸 사용
    if "fc.weight" in sd and sd["fc.weight"].ndim == 2:
        return sd["fc.weight"].shape[0]
    # 백업: 2D weight 중 가장 마지막 후보를 사용(모델에 따라 다를 수 있어 참고용)
    out = None
    for k, v in sd.items():
        if k.endswith("weight") and v.ndim == 2:
            out = v.shape[0]
    return out

ckpt_out_features = detect_out_features(state_dict)
if ckpt_out_features is not None:
    print("ckpt out_features =", ckpt_out_features)
    if ckpt_out_features != num_classes:
        print(f"⚠️ 경고: ckpt out_features({ckpt_out_features}) != dataset num_classes({num_classes})")
        # 필요 시 여기서 강제 종료하거나, 라벨/모델을 일치시키도록 조정하세요.

# ======================
# 4) 모델 생성 및 가중치 로드 (DataParallel 대비)
# ======================
model = CMTClassifier(num_classes).to(device).eval()

try:
    model.load_state_dict(state_dict, strict=True)
    print("✅ strict=True 로드 성공")
except RuntimeError as e:
    print("strict=True 실패 → module. 접두사 제거 후 로드 시도:", e)
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(cleaned, strict=False)
    print("✅ strict=False 로드 성공 (module. 제거)")

# ======================
# 5) 전처리 (학습 시 val_transform과 동일)
# ======================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# ======================
# 6) 안전한 이미지 로더 (PIL → 실패 시 OpenCV 우회)
# ======================
def load_image_rgb(path: str) -> Image.Image:
    # 1) PIL 시도
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e_pil:
        # 2) OpenCV 우회 (경로 인코딩/일부 코덱 문제 해결)
        if cv2 is None:
            raise e_pil
        data = np.fromfile(path, dtype=np.uint8)   # Windows 경로 안전
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise e_pil   # 실제로 손상/미지원 포맷
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

# ======================
# 7) 예측 함수
# ======================
@torch.inference_mode()
def predict_image(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")
    img = load_image_rgb(path)
    x = val_transform(img).unsqueeze(0).to(device)  # [1,3,224,224]
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    return label_names[idx], float(probs[idx].item())

# ======================
# 8) 실행
# ======================
try:
    label, prob = predict_image(TEST_IMAGE)
    print(f"예측 결과: {label} ({prob*100:.2f}%)")
except Exception as e:
    print("❌ 추론 실패:", e)
    # 추가 디버깅 힌트
    try:
        import pathlib
        p = pathlib.Path(TEST_IMAGE)
        print("exists:", p.exists(), " size:", p.stat().st_size if p.exists() else -1, " suffix:", p.suffix.lower())
    except Exception:
        pass
