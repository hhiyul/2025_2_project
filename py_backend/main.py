from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from model import VAL_TRANSFORM, load_cmt_model


logger = logging.getLogger(__name__)


class InferenceResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    prediction: str


class ModelService:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise ValueError("Label file must contain a JSON array of class names")
        if not labels:
            raise ValueError("Label file must contain at least one class name")

        model, missing_keys, unexpected_keys = load_cmt_model(
            self.model_path, num_classes=len(labels), device=self.device
        )
        if missing_keys:
            logger.warning("Missing keys when loading checkpoint: %s", sorted(missing_keys))
        if unexpected_keys:
            logger.warning("Unexpected keys when loading checkpoint: %s", sorted(unexpected_keys))

        self._model = model
        self._labels = list(labels)
        logger.info(
            "Model loaded successfully from %s (%d classes) on %s",
            self.model_path,
            len(self._labels),
            self.device,
        )

    def predict(self, image_bytes: bytes) -> Tuple[str, float]:
        self.ensure_loaded()
        assert self._model is not None, "Model must be loaded"
        assert self._labels, "Labels must be loaded"

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, index = probabilities.max(dim=1)

        label_index = int(index.item())
        confidence_value = float(confidence.item())
        if 0 <= label_index < len(self._labels):
            label = self._labels[label_index]
        else:
            logger.warning("Predicted label index %s outside label range", label_index)
            label = str(label_index)
        return label, confidence_value


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent


def _resolve_resource(name: str, *, env_var: str | None = None) -> Path:
    """프로젝트 디렉터리 안에 이미 존재하는 리소스를 찾아준다."""

    configured = os.getenv(env_var) if env_var else None
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        return (PROJECT_ROOT / candidate).resolve()

    relative = Path(name)
    search_order = [
        MODULE_DIR / relative,
        PROJECT_ROOT / relative,
    ]

    for candidate in search_order:
        if candidate.exists():
            return candidate

    # 존재하지 않더라도 가장 합리적인 위치(프로젝트 루트)를 반환해 이후 로딩 단계에서
    # FileNotFoundError를 명확하게 던지도록 한다.
    return (PROJECT_ROOT / relative).resolve()


MODEL_PATH = _resolve_resource("best_model.pt", env_var="MODEL_PATH")
LABELS_PATH = _resolve_resource("label_names.json", env_var="LABELS_PATH")

model_service = ModelService(MODEL_PATH, LABELS_PATH)
app = FastAPI(title="Inference Service")


@app.on_event("startup")
async def _startup_event() -> None:
    try:
        model_service.ensure_loaded()
    except Exception:  # pragma: no cover - startup failure should bubble up
        logger.exception("Failed to load model during startup")
        raise


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Inference service is running"}


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(file: UploadFile = File(...)) -> InferenceResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    size = len(contents)

    try:
        prediction, _ = model_service.predict(contents)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    return InferenceResponse(
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        size_bytes=size,
        prediction=prediction,
    )


@app.get("/hello/{name}")
async def say_hello(name: str) -> dict[str, str]:
    return {"message": f"Hello {name}"}
