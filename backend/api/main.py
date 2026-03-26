"""
FastAPI application entry point.

Serves:
  /api/...       — ML pipeline endpoints
  /              — Frontend (static HTML/JS/CSS)
  /health        — Health check

Models are loaded once at startup via lifespan context manager.
All heavy state lives in app.state to avoid module-level globals.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import predict, recourse
from backend.ml.classifier import CreditClassifier
from backend.ml.config import (
    ARTIFACTS_DIR,
    DICE_DATA_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_STATS_PATH,
    LABEL_ENCODERS_PATH,
    MAPIE_PATH,
    MODEL_PATH,
)
from backend.ml.counterfactuals import CounterfactualGenerator
from backend.ml.grey_zone import GreyZonePredictor

logger = logging.getLogger(__name__)

# ── Frontend path ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


# ── Lifespan: load models once on startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts…")

    try:
        app.state.classifier = CreditClassifier.load(MODEL_PATH)
        logger.info(f"Classifier loaded (val_auc={app.state.classifier.val_auc:.4f})")
    except FileNotFoundError:
        logger.warning("Model artifact not found. Run train.py first.")
        app.state.classifier = None

    try:
        app.state.grey_zone = GreyZonePredictor.load(MAPIE_PATH)
        logger.info("GreyZonePredictor loaded.")
    except FileNotFoundError:
        logger.warning("MAPIE artifact not found. Run train.py first.")
        app.state.grey_zone = None

    try:
        app.state.cf_generator = CounterfactualGenerator.load(DICE_DATA_PATH)
        logger.info("CounterfactualGenerator loaded.")
    except FileNotFoundError:
        logger.warning("DiCE artifact not found. Run train.py first.")
        app.state.cf_generator = None

    try:
        app.state.feature_names = joblib.load(FEATURE_NAMES_PATH)
        app.state.feature_stats = joblib.load(FEATURE_STATS_PATH)
        app.state.label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    except FileNotFoundError:
        app.state.feature_names = []
        app.state.feature_stats = {}
        app.state.label_encoders = {}

    logger.info("Startup complete.")
    yield

    logger.info("Shutting down…")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Recourse Engine",
    description=(
        "4-layer credit decision pipeline: "
        "XGBoost classifier → Conformal grey zone → "
        "Feasibility-constrained counterfactuals → Effort-ranked recourse paths"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(recourse.router, prefix="/api", tags=["recourse"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.classifier is not None,
        "grey_zone_loaded": app.state.grey_zone is not None,
        "cf_generator_loaded": app.state.cf_generator is not None,
    }


# ── Static files (frontend) ───────────────────────────────────────────────────
if (FRONTEND_DIR / "static").exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR / "static")),
        name="static",
    )


@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "templates" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Frontend not found. Check frontend/templates/index.html"})
