from __future__ import annotations

import contextlib

from config import settings
import uvicorn
from fastapi import FastAPI, HTTPException
from tensorflow.keras.saving import load_model

from filters import GaussianFilter
from pipeline import run_imaging_pipeline
from schemas import ImagingRequest

mlp_model = None
fcd_model = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global mlp_model, fcd_model
    mlp_path = settings.MODELS_DIR / settings.MLP_MODEL
    fcd_path = settings.MODELS_DIR / settings.FCD_MODEL
    try:
        mlp_model = load_model(str(mlp_path), compile=False)
        print(f"[startup] MLP loaded: {mlp_path}", flush=True)
        print(f"[startup] MLP input shape: {mlp_model.input_shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] MLP load failed: {exc}", flush=True)
        mlp_model = None
    try:
        fcd_model = load_model(
            str(fcd_path),
            custom_objects={"GaussianFilter": GaussianFilter},
            compile=False,
        )
        print(f"[startup] FCD loaded: {fcd_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] FCD load failed: {exc}", flush=True)
        fcd_model = None
    yield


app = FastAPI(
    title="STIX FCD imaging service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"mlp_loaded": mlp_model is not None, "fcd_loaded": fcd_model is not None}


@app.post("/imaging")
def imaging(request: ImagingRequest):
    if mlp_model is None:
        raise HTTPException(status_code=503, detail="MLP model not loaded")
    try:
        return run_imaging_pipeline(
            request.l1_json,
            request.selection,
            mlp_model,
            fcd_model,
            user_hpc_x=request.user_hpc_x,
            user_hpc_y=request.user_hpc_y,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "app:app", host="0.0.0.0", port=settings.PORT, reload=True
    )  # hot reload on for debugging
