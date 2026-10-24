"""FastAPI service exposing fraud-scoring endpoints.

Run from the project root with: uvicorn app.api:app --reload
"""

from pathlib import Path
import json

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import FraudNet


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(
    title="DeepGuard AI",
    description="Deep learning fraud risk scoring MVP",
    version="0.1.0",
)

model = None
scaler = None
metadata = None


class TransactionRequest(BaseModel):
    features: dict[str, float]


@app.on_event("startup")
def load_artifacts():
    global model, scaler, metadata

    required = [
        ARTIFACTS / "fraud_model.pt",
        ARTIFACTS / "scaler.joblib",
        ARTIFACTS / "metadata.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing model artifacts: {missing}. Run python -m app.train first."
        )

    with open(ARTIFACTS / "metadata.json") as f:
        metadata = json.load(f)

    # Loading a model checkpoint created by this project only.
    checkpoint = torch.load(
        ARTIFACTS / "fraud_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = FraudNet(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(ARTIFACTS / "scaler.joblib")


@app.get("/")
def root():
    return {
        "app": "DeepGuard AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "not_ready",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    columns = metadata["feature_columns"]
    supplied = request.features

    missing = sorted(set(columns) - set(supplied))
    extra = sorted(set(supplied) - set(columns))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "extra_features": extra},
        )

    try:
        values = np.asarray(
            [float(supplied[c]) for c in columns], dtype=np.float64
        )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail="All feature values must be numeric"
        )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422, detail="Features must be finite numbers"
        )

    transformed = scaler.transform(
        values.reshape(1, -1)
    ).astype("float32")

    with torch.no_grad():
        logits = model(torch.from_numpy(transformed))
        score = float(torch.sigmoid(logits).item())

    threshold = float(metadata["threshold"])
    flagged = score >= threshold

    return {
        "risk_score": round(score, 6),
        "threshold": round(threshold, 6),
        "decision": "REVIEW" if flagged else "LOWER_RISK",
        "flagged_for_review": flagged,
        "model": "DeepGuard FraudNet v0.1",
        "disclaimer": "Prototype risk score, not a production banking decision.",
    }


@app.get("/metrics")
def metrics():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata["metrics"]

