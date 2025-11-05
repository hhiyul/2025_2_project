from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


class InferenceResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    prediction: str


app = FastAPI(title="Inference Service")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Inference service is running"}


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(file: UploadFile = File(...)) -> InferenceResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    size = len(contents)

    prediction = "cat" if size % 2 == 0 else "dog"

    return InferenceResponse(
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        size_bytes=size,
        prediction=prediction,
    )


@app.get("/hello/{name}")
async def say_hello(name: str) -> dict[str, str]:
    return {"message": f"Hello {name}"}
