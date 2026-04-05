"""
api/serve.py — Production Inference API
========================================
FastAPI application that serves the fraud detection model.

Endpoints
---------
POST /predict           - single claim prediction
POST /predict/batch     - batch predictions (up to 1000)
GET  /health            - health check + model metadata
GET  /metrics           - model performance summary

Run locally:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker build -t fraud-api .
    docker run -p 8000:8000 fraud-api
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import sys                                                         
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model.pkl"
META_PATH  = Path(__file__).parent.parent / "models" / "model_meta.json"


# ─────────────────────────────── Pydantic schemas ──────────────────────────
# These define the exact shape of requests and responses.
# FastAPI uses them to validate input and auto-generate the /docs page.

class ClaimInput(BaseModel):
    Month: str
    WeekOfMonth: int = Field(..., ge=1, le=5)
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int = Field(..., ge=1, le=5)
    Sex: str
    MaritalStatus: str
    Age: int = Field(..., ge=0, le=120)
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    PolicyNumber: Optional[int] = None
    RepNumber: Optional[int] = None
    Deductible: int
    DriverRating: int = Field(..., ge=1, le=4)
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str
    AddressChange_Claim: str
    NumberOfCars: str
    Year: Optional[int] = None
    BasePolicy: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_predicted: bool
    risk_tier: str       # LOW / MEDIUM / HIGH / CRITICAL
    confidence: str      # LOW / MEDIUM / HIGH
    model_version: str
    inference_ms: float


class BatchRequest(BaseModel):
    claims: list[ClaimInput]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_claims: int
    flagged_count: int
    inference_ms: float


# ─────────────────────────────── Model loader ──────────────────────────────

class FraudModelServer:
    """Wraps the sklearn Pipeline with lazy loading and helper methods."""

    def __init__(self):
        self._pipeline  = None
        self._meta      = None

    def load(self):
        logger.info(f"Loading model from {MODEL_PATH}")
        self._pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self._meta = json.load(f)
        logger.info(f"Model loaded — ROC-AUC: {self._meta.get('cv_roc_auc', 'N/A')}")

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.load()
        return self._pipeline

    @property
    def threshold(self) -> float:
        if self._meta is None:
            self.load()
        return self._meta["threshold"]

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta

    def predict_single(self, claim_dict: dict) -> dict:
        start = time.perf_counter()
        df    = pd.DataFrame([claim_dict])
        prob  = float(self.pipeline.predict_proba(df)[0, 1])
        return {
            "fraud_probability": round(prob, 4),
            "fraud_predicted":   prob >= self.threshold,
            "risk_tier":         self._risk_tier(prob),
            "confidence":        self._confidence(prob),
            "model_version":     self.meta.get("model_type", "unknown"),
            "inference_ms":      round((time.perf_counter() - start) * 1000, 2),
        }

    def predict_batch(self, claims: list[dict]) -> dict:
        start  = time.perf_counter()
        df     = pd.DataFrame(claims)
        probs  = self.pipeline.predict_proba(df)[:, 1]
        preds  = probs >= self.threshold
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "predictions": [
                {
                    "fraud_probability": round(float(p), 4),
                    "fraud_predicted":   bool(pred),
                    "risk_tier":         self._risk_tier(float(p)),
                    "confidence":        self._confidence(float(p)),
                    "model_version":     self.meta.get("model_type", "unknown"),
                    "inference_ms":      round(elapsed_ms / len(claims), 2),
                }
                for p, pred in zip(probs, preds)
            ],
            "total_claims":  len(claims),
            "flagged_count": int(preds.sum()),
            "inference_ms":  elapsed_ms,
        }

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob < 0.10: return "LOW"
        if prob < 0.30: return "MEDIUM"
        if prob < 0.60: return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _confidence(prob: float) -> str:
        distance = abs(prob - 0.3)
        if distance > 0.35: return "HIGH"
        if distance > 0.15: return "MEDIUM"
        return "LOW"


# ─────────────────────────────── FastAPI app ───────────────────────────────

model_server = FraudModelServer()

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Real-time fraud scoring for insurance claims",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model into memory once at startup — not on first request."""
    model_server.load()


@app.get("/health")
async def health():
    """Returns model status and metadata. Used by Docker HEALTHCHECK."""
    return {
        "status":       "healthy",
        "model_loaded": model_server._pipeline is not None,
        "model_meta":   model_server.meta,
    }

@app.get("/ping")
async def ping():
    """SageMaker health check endpoint — must return 200."""
    return {"status": "healthy"}

@app.post("/invocations")
async def invocations(claim: ClaimInput):
    """SageMaker prediction endpoint — maps to /predict."""
    try:
        return model_server.predict_single(claim.model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Returns CV metrics and training metadata."""
    return model_server.meta


@app.post("/predict", response_model=PredictionResponse)
async def predict(claim: ClaimInput):
    """Score a single insurance claim for fraud probability."""
    try:
        return model_server.predict_single(claim.model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Score up to 1000 claims in a single request."""
    if len(request.claims) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 claims per batch")
    try:
        return model_server.predict_batch([c.model_dump() for c in request.claims])
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
