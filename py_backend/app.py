from __future__ import annotations

import io
import json
import os
from pathlib import Path
from threading import Lock
from typing import Tuple
from fastapi.security.api_key import APIKeyHeader

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError


from .models import VAL_TRANSFORM, load_cmt_model


BASE_DIR = Path(__file__).resolve().parent

def _path_from_env_or_default(env_var: str, *relative: str) -> Path:
    v = os.getenv(env_var)
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p.resolve()
    return (BASE_DIR.joinpath(*relative)).resolve()

MODEL_PATH  = _path_from_env_or_default("MODEL_PATH",  "model_pt", "best_model_fold1.pt")
LABELS_PATH = _path_from_env_or_default("LABELS_PATH", "model_pt", "label_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Label file not found: {LABELS_PATH}")


class InferenceResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    prediction: str
    confidence: float



# ModelService
# ----------------------------
class ModelService:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = DEVICE
        self.transform = VAL_TRANSFORM
        self._model: torch.nn.Module | None = None
        self._labels: list[str] = []
        self._lock = Lock()

    def ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Label file not found: {self.labels_path}")

        with self.labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list) or not all(isinstance(s, str) for s in labels) or not labels:
            raise ValueError("Label file must be a non-empty JSON array of strings")

        model, missing, unexpected = load_cmt_model(
            model_path=self.model_path, num_classes=len(labels), device=self.device
        )
        if missing:
            print("[load] missing keys:", list(missing))
        if unexpected:
            print("[load] unexpected keys:", list(unexpected))

        self._model = model.to(self.device).eval()
        self._labels = list(labels)
        torch.set_num_threads(1)  # CPU 서버면 과한 스레드 방지(상황 맞게 조절)

    def predict(self, image_bytes: bytes) -> Tuple[str, float]:
        self.ensure_loaded()
        assert self._model is not None
        assert self._labels

        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self._model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, idx = probs.max(dim=0)
        label = self._labels[int(idx)]
        return label, float(conf)


# ----------------------------
# FastAPI 앱 & 라우트
# ----------------------------
app = FastAPI(
    title="융소프",
    description="딥러닝이에요")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model_service = ModelService(MODEL_PATH, LABELS_PATH)

#API키
API_KEY = "fuck-key-123"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

#api키 맞나 체크하는거 틀리면 오류코드 반환
async def check_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

@app.get("/", summary="홈")
def root():
    return {"message": "정상작동 중"}


@app.get("/health", summary="연결상태확인", dependencies=[Depends(check_api_key)])
def health():
    """
    api 연결상태 정상인지 확인하는 기능
    """
    return {
        "model_loaded": model_service._model is not None,
        "labels_loaded": bool(model_service._labels),
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
    }


# ✅ 시각테스트용 UI 다 만들면 없앨거임
@app.get("/ui", summary="시각용ui", response_class=HTMLResponse)
def ui():
    """
    테스트용 뒤에서 돌아가는지 시각화함
    """
    return HTMLResponse(
        """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>Inference UI</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 30px; }
    .card { max-width: 520px; padding: 20px; border: 1px solid #ddd; border-radius: 12px; }
    .row { margin-top: 12px; }
    img { max-width: 100%; border-radius: 8px; }
    button { padding: 10px 14px; border-radius: 8px; border: 1px solid #ccc; cursor:pointer; }
    #result { margin-top: 10px; font-weight: 600; }
    #err { color: #b00020; margin-top: 8px; }
  </style>
</head>
<body>
  <h2>이미지 분류 테스트</h2>
  <div class="card">
    <div class="row">
      <input id="file" type="file" accept="image/*"/>
    </div>
    <div class="row">
      <img id="preview" alt="preview" />
    </div>
    <div class="row">
      <button id="btn">분류 요청</button>
    </div>
    <div id="result"></div>
    <div id="err"></div>
  </div>

<script>
const $ = id => document.getElementById(id);
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  $("preview").src = url;
});
$("btn").addEventListener("click", async () => {
  $("result").textContent = "";
  $("err").textContent = "";
  const f = $("file").files[0];
  if (!f) { $("err").textContent = "이미지를 선택하세요."; return; }
  const form = new FormData();
  form.append("file", f);
  try {
    const res = await fetch("/infer", { method: "POST", body: form });
    if (!res.ok) {
      const msg = await res.text();
      $("err").textContent = "오류: " + msg;
      return;
    }
    const data = await res.json();
    $("result").textContent = `예측: ${data.prediction}  (conf: ${(data.confidence*100).toFixed(1)}%)`;
  } catch (e) {
    $("err").textContent = "요청 실패: " + e;
  }
});
</script>
</body>
</html>
        """.strip()
    )


@app.post("/infer", summary="딥러닝추론", response_model=InferenceResponse, dependencies=[Depends(check_api_key)])
async def infer(file: UploadFile = File(...)):
    """
    딥러닝 추론 api
    """
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    pred, conf = model_service.predict(blob)
    return InferenceResponse(
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        size_bytes=len(blob),
        prediction=pred,
        confidence=conf,
    )
