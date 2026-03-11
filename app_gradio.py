# app_gradio.py
"""
Gradio demo wrapper for your demand prediction model.
Place this file in the same directory as:
  - demand_model_calibrated.pkl
  - le_item.pkl
  - feature_names.pkl

When HF builds the Space it will run this app.
"""

import joblib
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime

# Paths (must exist in repo)
CALIBRATED_MODEL_PATH = "demand_model_calibrated.pkl"
ENCODER_PATH = "le_item.pkl"
FEATURES_PATH = "feature_names.pkl"

# Load artifacts
_model = joblib.load(CALIBRATED_MODEL_PATH)
_le_item = joblib.load(ENCODER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

SEASON_MAP = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

def _make_features(product_identifier, dt_iso):
    # robust datetime parse
    if isinstance(dt_iso, (int, float)):
        dt = datetime.fromtimestamp(dt_iso)
    else:
        dt = pd.to_datetime(dt_iso, errors="coerce")
        if pd.isna(dt):
            raise ValueError("Could not parse datetime")
        dt = dt.to_pydatetime()

    # item encoding (accept original itemDescription or encoded int)
    if isinstance(product_identifier, str):
        try:
            item_enc = int(_le_item.transform([product_identifier])[0])
        except Exception:
            item_enc = int(_le_item.transform([product_identifier.strip().title()])[0])
    else:
        item_enc = int(product_identifier)

    month = dt.month; day = dt.day; dow = dt.weekday()
    week_of_year = int(dt.isocalendar()[1])
    try:
        quarter = dt.quarter
    except Exception:
        quarter = ((month - 1)//3) + 1

    is_weekend = 1 if dow >= 5 else 0
    is_friday  = 1 if dow == 4 else 0
    is_monday  = 1 if dow == 0 else 0
    is_month_start = 1 if day <= 3 else 0
    is_month_end   = 1 if day >= 28 else 0
    is_mid_month   = 1 if (14 <= day <= 16) else 0

    season = SEASON_MAP.get(month, 0)
    is_christmas_period = 1 if ((month == 12 and day >= 18) or (month == 1 and day <= 2)) else 0
    is_summer_holiday   = 1 if (7 <= month <= 8) else 0

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
    ordered = {f: feat[f] for f in FEATURES if f in feat}
    return ordered

def predict(product_id, dt_str):
    try:
        feats = _make_features(product_id, dt_str)
    except Exception as e:
        return {"error": str(e)}
    df = pd.DataFrame([feats])
    probs = _model.predict_proba(df)[0]
    classes = _model.classes_
    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
    top = str(classes[np.argmax(probs)])
    return {"probabilities": prob_map, "predicted_label": top, "input_features": feats}

# Build UI
with gr.Blocks() as demo:
    gr.Markdown("# Demand prediction — Dynamic Pricing PoC")
    with gr.Row():
        pid = gr.Textbox(label="product_id (itemDescription or encoded int)", value="")
        dt = gr.Textbox(label="datetime (ISO, e.g. 2026-03-21T12:00:00)", value=datetime.now().isoformat(timespec="seconds"))
    btn = gr.Button("Predict")
    out = gr.JSON()
    btn.click(fn=predict, inputs=[pid, dt], outputs=[out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)