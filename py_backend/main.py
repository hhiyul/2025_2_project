from __future__ import annotations

import logging
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile

from pydantic import BaseModel


#uvicorn py_backend.app:app --host 0.0.0.0 --port 8000 --reload
#ngrok http 8000
#위에 2줄은 무시해도됨
logger = logging.getLogger(__name__)


class InferenceResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    prediction: str


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

app = FastAPI(title="Inference Service")
