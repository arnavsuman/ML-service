import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import warnings
import time

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# app.py
"""
FastAPI service for your demand prediction model.
Place this file in the same directory as:
  - demand_model_calibrated.pkl
  - le_item.pkl
  - feature_names.pkl

Run: python app.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import joblib
from datetime import datetime
import uvicorn
import os

# -------- CONFIG: artifact filenames (must exist in this folder) --------
CALIBRATED_MODEL_PATH = "demand_model_calibrated.pkl"
ENCODER_PATH           = "le_item.pkl"
FEATURES_PATH          = "feature_names.pkl"
# -----------------------------------------------------------------------

# Quick sanity check
for p in (CALIBRATED_MODEL_PATH, ENCODER_PATH, FEATURES_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required artifact not found: {p}")

# Load artifacts once at cold-start
_model = joblib.load(CALIBRATED_MODEL_PATH)   # calibrated classifier (sklearn pipeline)
_le_item = joblib.load(ENCODER_PATH)         # LabelEncoder for itemDescription
FEATURES = joblib.load(FEATURES_PATH)        # list of feature names used for training

app = FastAPI(title="Demand Predictor (Dynamic Pricing PoC)")

# Pydantic models for validated input
class PredictRequest(BaseModel):
    product_id: Union[str, int]
    datetime: str  # ISO-8601 string or many common date formats

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]

# Helper: season map used during training
SEASON_MAP = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}

def _make_features(product_identifier, dt_iso):
    """
    Build the same features used in training.
    Accepts either the original itemDescription string (preferred),
    or the numeric item_enc (LabelEncoder value).
    dt_iso: ISO datetime string or epoch (int/float)
    Returns: ordered dict of features matching FEATURES list.
    """
    # parse datetime robustly
    if isinstance(dt_iso, (int, float)):
        dt = datetime.fromtimestamp(dt_iso)
    else:
        dt = pd.to_datetime(dt_iso, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Unable to parse datetime: {dt_iso}")
        dt = dt.to_pydatetime()

    # item encoding: accept string name or integer encoded value
    if isinstance(product_identifier, str):
        # try exact transform (LabelEncoder needs the exact original string)
        try:
            item_enc = int(_le_item.transform([product_identifier])[0])
        except Exception:
            # fallback: mimic training normalization (strip + title)
            try:
                item_enc = int(_le_item.transform([product_identifier.strip().title()])[0])
            except Exception as e:
                raise ValueError(f"Unknown product_id string: '{product_identifier}'. Provide the original itemDescription or an encoded integer.") from e
    elif isinstance(product_identifier, (int, np.integer)):
        item_enc = int(product_identifier)
    else:
        # attempt cast
        item_enc = int(product_identifier)

    month = dt.month
    day = dt.day
    dow = dt.weekday()
    week_of_year = int(dt.isocalendar()[1])
    # quarter may be attribute on pandas.Timestamp; safe fallback:
    try:
        quarter = dt.quarter
    except Exception:
        quarter = ((month - 1) // 3) + 1

    is_weekend = 1 if dow >= 5 else 0
    is_friday  = 1 if dow == 4 else 0
    is_monday  = 1 if dow == 0 else 0
    is_month_start = 1 if day <= 3 else 0
    is_month_end   = 1 if day >= 28 else 0
    is_mid_month   = 1 if (14 <= day <= 16) else 0

    season = SEASON_MAP.get(month, 0)
    is_christmas_period = 1 if ((month == 12 and day >= 18) or (month == 1 and day <= 2)) else 0
    is_summer_holiday   = 1 if (month >= 7 and month <= 8) else 0

    dow_month_interaction = dow * 12 + month
    weekend_season = is_weekend * season
    month_start_weekend = is_month_start * is_weekend

    feat = {
        "item_enc": item_enc,
        "day_of_week": dow,
        "month": month,
        "quarter": quarter,
        "week_of_year": week_of_year,
        "day_of_month": day,
        "season": season,
        "is_weekend": is_weekend,
        "is_friday": is_friday,
        "is_monday": is_monday,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,
        "is_mid_month": is_mid_month,
        "is_christmas_period": is_christmas_period,
        "is_summer_holiday": is_summer_holiday,
        "dow_month_interaction": dow_month_interaction,
        "weekend_season": weekend_season,
        "month_start_weekend": month_start_weekend,
    }

    # Keep only and order features exactly as in FEATURES
    ordered = {f: feat[f] for f in FEATURES if f in feat}
    # Check for any missing features
    missing = [f for f in FEATURES if f not in ordered]
    if missing:
        raise RuntimeError(f"Missing expected features: {missing}")
    return ordered

def _format_output(probs, classes):
    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
    top_idx = int(np.argmax(probs))
    return {
        "probabilities": prob_map,
        "predicted_label": str(classes[top_idx]),
        "confidence": float(probs[top_idx])
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Single prediction.
    Request JSON: {"product_id": "Whole Milk", "datetime": "2026-03-21T12:00:00"}
    or product_id can be the integer label (item_enc).
    """
    try:
        feats = _make_features(req.product_id, req.datetime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = pd.DataFrame([feats])
    probs = _model.predict_proba(df)[0]
    classes = _model.classes_
    out = _format_output(probs, classes)
    out["input_features"] = feats
    return out

@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    """
    Batch prediction.
    Request JSON: {"items": [{"product_id":"A","datetime":"2026-03-21T12:00:00"}, ...]}
    """
    results = []
    for item in req.items:
        try:
            feats = _make_features(item.product_id, item.datetime)
            df = pd.DataFrame([feats])
            probs = _model.predict_proba(df)[0]
            classes = _model.classes_
            out = _format_output(probs, classes)
            out["input_features"] = feats
            results.append(out)
        except Exception as e:
            results.append({"error": str(e), "item": {"product_id": item.product_id, "datetime": item.datetime}})
    return {"predictions": results}

if __name__ == "__main__":
    # run with: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")