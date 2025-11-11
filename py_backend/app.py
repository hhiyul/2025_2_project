from __future__ import annotations

import io
import json
import os
from pathlib import Path
from threading import Lock
from typing import Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

# 🔸 너의 모델 코드 파일명이 py_backend/models.py 라면 아래처럼 상대 임포트로 가져와.
from .models import VAL_TRANSFORM, load_cmt_model


# ----------------------------
# 경로 유틸 (프로젝트 루트/리소스 찾기)
# ----------------------------
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

def _resolve_resource(name: str, *, env_var: str | None = None) -> Path:
    """프로젝트 내 리소스 경로를 (환경변수 > 모듈 디렉토리 > 루트) 우선순위로 찾는다."""
    configured = os.getenv(env_var) if env_var else None
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    relative = Path(name)
    for candidate in (MODULE_DIR / relative, PROJECT_ROOT / relative):
        if candidate.exists():
            return candidate
    # 존재하지 않아도 합리적 기본 경로 반환(이후 단계에서 FileNotFoundError로 명확히 터뜨림)
    return (PROJECT_ROOT / relative).resolve()


# ----------------------------
# 설정 (환경변수로 오버라이드 가능)
# ----------------------------
# 예) PowerShell:
#   $env:MODEL_PATH="...\2025_2_project\models\best_model_fold1.pt"
#   $env:LABELS_PATH="...\2025_2_project\models\labels.json"
MODEL_PATH  = _resolve_resource("models/best_model_fold1.pt", env_var="MODEL_PATH")
LABELS_PATH = _resolve_resource("models/labels.json",        env_var="LABELS_PATH")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Pydantic 응답 스키마
# ----------------------------
class InferenceResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    prediction: str
    confidence: float


# ----------------------------
# ModelService (service 분리 없이 여기 포함)
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
app = FastAPI(title="Inference Service (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model_service = ModelService(MODEL_PATH, LABELS_PATH)


@app.get("/")
def root():
    return {"message": "Inference service is running"}


@app.get("/health")
def health():
    return {
        "model_loaded": model_service._model is not None,
        "labels_loaded": bool(model_service._labels),
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
    }


# ✅ 시각테스트용 UI 다 만들면 없앨거임
@app.get("/ui", response_class=HTMLResponse)
def ui():
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


@app.post("/infer", response_model=InferenceResponse)
async def infer(file: UploadFile = File(...)):
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
