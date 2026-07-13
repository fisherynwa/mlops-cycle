"""Serve the current @champion GAM over HTTP.

Loads the model by *alias* (models:/insurance-cost@champion) at startup, so
whenever you promote a new champion in training, restarting the server picks it
up — no code change, no version numbers hardcoded.

Run:
uvicorn src.serve:app --reload

Then open http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from loguru import logger
from src.config import (MLFLOW_URI, MODEL_URI, JSD_THRESHOLD, NUM_COLS, CAT_COLS, TARGET, ENCODERS)
from pydantic import BaseModel, Field


# smoker encoding must match training: category codes are alphabetical -> no=0, yes=1

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info("Loading model from {}", MODEL_URI)
    _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    mv = mlflow.MlflowClient().get_model_version_by_alias("insurance-cost", "champion")
    _state["version"] = mv.version
    logger.success("Champion v{} loaded", mv.version)
    yield
    _state.clear()


app = FastAPI(title="insurance-cost model", lifespan=lifespan)

# --- Prometheus metrics ---
Instrumentator().instrument(app).expose(app)   # auto /metrics: request count, latency, status
PRED_CHARGE = Histogram("prediction_charge", "Predicted insurance charge",
                        buckets=[5000, 10000, 20000, 30000, 40000, 60000])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["model_version"])

# pydantic models for request and response bodies, with validation and examples
class ClaimFeatures(BaseModel):
    age: int = Field(ge=0, le=120, examples=[45])
    bmi: float = Field(ge=10, le=60, examples=[30.5])
    smoker: Literal["yes", "no"] = Field(examples=["yes"])


class Prediction(BaseModel):
    charge: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model-info")
def model_info() -> dict:
    return {"model_uri": MODEL_URI, "champion_version": _state.get("version")}


@app.post("/predict", response_model=Prediction)
def predict(x: ClaimFeatures) -> Prediction:
    row = pd.DataFrame([x.model_dump()])
    for col, mapping in ENCODERS.items():          # encode every categorical from config
        row[col] = row[col].map(mapping)
    row = row[NUM_COLS + CAT_COLS]                  # column order must match training
    charge = float(_state["model"].predict(row)[0])
    PRED_CHARGE.observe(charge)
    PREDICTIONS.labels(model_version=str(_state["version"])).inc()
    return Prediction(charge=round(charge, 2), model_version=str(_state["version"]))