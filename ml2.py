# """
# Dynamic Pricing — Demand Prediction Model
# Hardware target : Apple M1 Max, 32GB RAM
# Model           : LightGBM + Optuna hyperparameter tuning
# Expected runtime: 10–20 minutes
# """

# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import optuna
# import joblib
# import warnings
# import time
# warnings.filterwarnings("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# from sklearn.model_selection  import StratifiedKFold, cross_val_score, train_test_split
# from sklearn.preprocessing    import LabelEncoder
# from sklearn.metrics          import classification_report, confusion_matrix, f1_score
# from sklearn.calibration      import CalibratedClassifierCV

# # ──────────────────────────────────────────────
# #  CONFIG
# # ──────────────────────────────────────────────
# INPUT_FILE     = "Groceries_dataset.csv"
# MODEL_OUT      = "demand_model.pkl"
# ENCODER_OUT    = "le_item.pkl"
# CALIBRATED_OUT = "demand_model_calibrated.pkl"
# OPTUNA_TRIALS  = 80    # ~15–20 mins on M1 Max; increase to 150 for best results
# CV_FOLDS       = 10
# RANDOM_STATE   = 42
# # ──────────────────────────────────────────────

# print("=" * 60)
# print("  DEMAND MODEL TRAINING PIPELINE")
# print("=" * 60)


# # ── 1. Load & parse dates
# print("\n[1/7] Loading data ...")
# df = pd.read_csv(INPUT_FILE)
# df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
# df = df.dropna(subset=["Date"])
# df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.title()
# df = df[df["itemDescription"] != "Nan"]
# print(f"      {len(df):,} rows loaded")


# # ── 2. Rich feature engineering
# print("\n[2/7] Engineering features ...")

# df["day_of_week"]    = df["Date"].dt.dayofweek
# df["month"]          = df["Date"].dt.month
# df["day_of_month"]   = df["Date"].dt.day
# df["week_of_year"]   = df["Date"].dt.isocalendar().week.astype(int)
# df["quarter"]        = df["Date"].dt.quarter
# df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
# df["is_friday"]      = (df["day_of_week"] == 4).astype(int)   # pre-weekend spike
# df["is_monday"]      = (df["day_of_week"] == 0).astype(int)   # post-weekend drop
# df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)  # payday effect
# df["is_month_end"]   = (df["day_of_month"] >= 28).astype(int)
# df["is_mid_month"]   = (
#     (df["day_of_month"] >= 14) & (df["day_of_month"] <= 16)
# ).astype(int)

# df["season"] = df["month"].map({
#     12:0, 1:0, 2:0,   # Winter
#     3:1,  4:1, 5:1,   # Spring
#     6:2,  7:2, 8:2,   # Summer
#     9:3, 10:3, 11:3   # Autumn
# })

# # Holiday proximity (no library needed)
# df["is_christmas_period"] = (
#     ((df["month"] == 12) & (df["day_of_month"] >= 18)) |
#     ((df["month"] ==  1) & (df["day_of_month"] <=  2))
# ).astype(int)
# df["is_summer_holiday"] = (
#     (df["month"] >= 7) & (df["month"] <= 8)
# ).astype(int)

# # Slot demand — how many purchases of this item on this day_of_week + month
# df["slot_count"] = df.groupby(
#     ["itemDescription", "day_of_week", "month"]
# )["Date"].transform("count")

# # Per-item baseline stats
# item_stats = (
#     df.groupby("itemDescription")["slot_count"]
#       .agg(["mean","std","median","min","max"])
#       .reset_index()
# )
# item_stats.columns = [
#     "itemDescription","item_mean","item_std",
#     "item_median","item_min","item_max"
# ]
# item_stats["item_std"]   = item_stats["item_std"].fillna(1).replace(0, 1)
# item_stats["item_range"] = item_stats["item_max"] - item_stats["item_min"]
# df = df.merge(item_stats, on="itemDescription", how="left")

# # Z-score — how unusual is this slot relative to this item's baseline
# df["z_score"] = (df["slot_count"] - df["item_mean"]) / df["item_std"]

# # Normalised slot count 0–1 within each item
# df["slot_count_norm"] = (df["slot_count"] - df["item_min"]) / (
#     df["item_range"].replace(0, 1)
# )

# # Interaction features
# df["dow_month_interaction"] = df["day_of_week"] * 12 + df["month"]  # captures Dec Saturdays
# df["weekend_season"]        = df["is_weekend"] * df["season"]
# df["month_start_weekend"]   = df["is_month_start"] * df["is_weekend"]

# # Item global popularity percentile rank
# pop = df.groupby("itemDescription").size().rank(pct=True)
# df["item_popularity_pct"] = df["itemDescription"].map(pop)

# print(f"      Features engineered ✅")


# # ── 3. Create demand label using z-score
# print("\n[3/7] Creating demand labels ...")

# def label(z):
#     if   z >= 0.5:  return "HIGH"
#     elif z >= -0.5: return "NORMAL"
#     else:           return "LOW"

# df["demand_label"] = df["z_score"].apply(label)
# dist = df["demand_label"].value_counts()
# for lbl in ["HIGH","NORMAL","LOW"]:
#     cnt = dist.get(lbl, 0)
#     print(f"      {lbl:<8} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")


# # ── 4. Encode & prepare
# print("\n[4/7] Encoding features ...")

# le = LabelEncoder()
# df["item_enc"] = le.fit_transform(df["itemDescription"])

# FEATURES = [
#     "item_enc",
#     "day_of_week", "month", "quarter", "week_of_year", "day_of_month",
#     "season",
#     "is_weekend", "is_friday", "is_monday",
#     "is_month_start", "is_month_end", "is_mid_month",
#     "is_christmas_period", "is_summer_holiday",
#     "slot_count", "slot_count_norm", "z_score",
#     "item_mean", "item_std", "item_median", "item_range", "item_popularity_pct",
#     "dow_month_interaction", "weekend_season", "month_start_weekend",
# ]

# X = df[FEATURES].copy()
# y = df["demand_label"]
# print(f"      {len(FEATURES)} features × {len(X):,} rows")


# # ── 5. Train / val split
# X_tr, X_val, y_tr, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
# )


# # ── 6. Optuna hyperparameter search
# print(f"\n[5/7] Optuna search — {OPTUNA_TRIALS} trials, grab a coffee ☕ ...")
# start = time.time()

# def objective(trial):
#     # "goss" removed — it does not support subsample/bagging params
#     boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

#     params = {
#         "objective":         "multiclass",
#         "num_class":         3,
#         "metric":            "multi_logloss",
#         "verbosity":         -1,
#         "boosting_type":     boosting_type,
#         "num_leaves":        trial.suggest_int("num_leaves", 64, 1024),
#         "max_depth":         trial.suggest_int("max_depth", 6, 24),
#         "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#         "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#         "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
#         "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
#         "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
#         "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
#         "max_bin":           trial.suggest_int("max_bin", 128, 512),
#         "class_weight":      "balanced",
#         "random_state":      RANDOM_STATE,
#         "n_jobs":            -1,
#     }

#     # subsample (bagging) is only valid for gbdt
#     # dart uses dropout instead of bagging so these params are ignored/invalid
#     if boosting_type == "gbdt":
#         params["subsample"]      = trial.suggest_float("subsample", 0.5, 1.0)
#         params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 7)

#     model = lgb.LGBMClassifier(**params)
#     model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         callbacks=[
#             lgb.early_stopping(50, verbose=False),
#             lgb.log_evaluation(-1)
#         ]
#     )
#     preds = model.predict(X_val)
#     return f1_score(y_val, preds, average="weighted")

# study = optuna.create_study(
#     direction="maximize",
#     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
#     pruner=optuna.pruners.MedianPruner(n_warmup_steps=15)
# )
# # catch=(Exception,) → failed trials are skipped not crash the whole run
# study.optimize(objective, n_trials=OPTUNA_TRIALS,
#                show_progress_bar=True, catch=(Exception,))

# elapsed = time.time() - start
# print(f"\n      ✅ Tuning complete in {elapsed/60:.1f} mins")
# print(f"      Best weighted F1 : {study.best_value:.4f}")
# print(f"      Best params:")
# for k, v in study.best_params.items():
#     print(f"        {k:<25} {v}")


# # ── 7. Train final model with best params
# print("\n[6/7] Training final model + cross-validation ...")

# best_params = {
#     **study.best_params,
#     "objective":    "multiclass",
#     "num_class":    3,
#     "metric":       "multi_logloss",
#     "class_weight": "balanced",
#     "random_state": RANDOM_STATE,
#     "n_jobs":       -1,
#     "verbosity":    -1,
# }

# final_model = lgb.LGBMClassifier(**best_params)
# final_model.fit(X_tr, y_tr)

# # 10-fold cross-validation for unbiased performance estimate
# print(f"      Running {CV_FOLDS}-fold stratified cross-validation ...")
# cv_model = lgb.LGBMClassifier(**best_params)
# cv_scores = cross_val_score(
#     cv_model, X, y,
#     cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
#     scoring="f1_weighted",
#     n_jobs=-1
# )
# print(f"\n      CV weighted F1 per fold : {[round(s,3) for s in cv_scores]}")
# print(f"      Mean  : {cv_scores.mean():.4f}")
# print(f"      Std   : {cv_scores.std():.4f}  (lower = more stable)")

# # Held-out evaluation
# y_pred = final_model.predict(X_val)
# print(f"\n      Held-out test report:")
# print(classification_report(y_val, y_pred, digits=3))

# print("      Confusion matrix:")
# cm = confusion_matrix(y_val, y_pred, labels=["HIGH","NORMAL","LOW"])
# cm_df = pd.DataFrame(cm,
#     index  =["Actual HIGH","Actual NORMAL","Actual LOW"],
#     columns=["Pred HIGH",  "Pred NORMAL",  "Pred LOW"]
# )
# print(cm_df.to_string())

# # Feature importance
# print("\n      Feature importances (split gain):")
# imp_series = pd.Series(
#     final_model.feature_importances_, index=FEATURES
# ).sort_values(ascending=False)
# for feat, imp in imp_series.items():
#     bar = "█" * int(imp / imp_series.max() * 35)
#     print(f"        {feat:<30} {imp:>7.0f}  {bar}")


# # ── 8. Calibrate probabilities & save
# print("\n[7/7] Calibrating & saving ...")

# # Calibration makes confidence scores trustworthy for the pricing engine
# # "87% confident HIGH" should actually be right ~87% of the time
# calibrated = CalibratedClassifierCV(final_model, cv=5, method="isotonic")
# calibrated.fit(X_tr, y_tr)

# joblib.dump(final_model, MODEL_OUT)
# joblib.dump(calibrated,  CALIBRATED_OUT)
# joblib.dump(le,          ENCODER_OUT)
# joblib.dump(FEATURES,    "feature_names.pkl")

# print(f"      ✅ {MODEL_OUT}")
# print(f"      ✅ {CALIBRATED_OUT}  ← use this in FastAPI")
# print(f"      ✅ {ENCODER_OUT}")
# print(f"      ✅ feature_names.pkl")


# # ── Inference example — how FastAPI will call this
# print("\n" + "=" * 60)
# print("  INFERENCE EXAMPLE")
# print("=" * 60)

# try:
#     item_row = df[df["itemDescription"] == "Whole Milk"].iloc[0]
#     sample = pd.DataFrame([{
#         "item_enc"             : le.transform(["Whole Milk"])[0],
#         "day_of_week"          : 5,
#         "month"                : 12,
#         "quarter"              : 4,
#         "week_of_year"         : 51,
#         "day_of_month"         : 21,
#         "season"               : 0,
#         "is_weekend"           : 1,
#         "is_friday"            : 0,
#         "is_monday"            : 0,
#         "is_month_start"       : 0,
#         "is_month_end"         : 0,
#         "is_mid_month"         : 0,
#         "is_christmas_period"  : 1,
#         "is_summer_holiday"    : 0,
#         "slot_count"           : item_row["item_mean"] * 1.5,
#         "slot_count_norm"      : 0.75,
#         "z_score"              : 0.8,
#         "item_mean"            : item_row["item_mean"],
#         "item_std"             : item_row["item_std"],
#         "item_median"          : item_row["item_median"],
#         "item_range"           : item_row["item_range"],
#         "item_popularity_pct"  : item_row["item_popularity_pct"],
#         "dow_month_interaction": 5 * 12 + 12,
#         "weekend_season"       : 1 * 0,
#         "month_start_weekend"  : 0,
#     }])

#     proba   = calibrated.predict_proba(sample)[0]
#     classes = calibrated.classes_
#     pred    = classes[np.argmax(proba)]
#     conf    = float(np.max(proba))

#     print(f"\n  Input  → Whole Milk | Saturday Dec 21 | Christmas period")
#     print(f"  Output → demand = {pred}  (confidence: {conf:.0%})")
#     print(f"  Probs  → { {c: f'{p:.1%}' for c, p in zip(classes, proba)} }")
# except Exception as e:
#     print(f"  (Inference example skipped: {e})")

# print("\n✅ Pipeline complete!")

## OUTPUT
# ============================================================
#   DEMAND MODEL TRAINING PIPELINE
# ============================================================

# [1/7] Loading data ...
#       23,509 rows loaded

# [2/7] Engineering features ...
#       Features engineered ✅

# [3/7] Creating demand labels ...
#       HIGH      6,829  (29.0%)
#       NORMAL    8,544  (36.3%)
#       LOW       8,136  (34.6%)

# [4/7] Encoding features ...
#       26 features × 23,509 rows

# [5/7] Optuna search — 80 trials, grab a coffee ☕ ...
# Best trial: 0. Best value: 1: 100%|█████████████████████████████████████████████████████████████████████████| 80/80 [15:48<00:00, 11.85s/it]

#       ✅ Tuning complete in 15.8 mins
#       Best weighted F1 : 1.0000
#       Best params:
#         boosting_type             dart
#         num_leaves                767
#         max_depth                 17
#         learning_rate             0.00889039845957559
#         n_estimators              565
#         min_child_samples         10
#         colsample_bytree          0.9197056874649611
#         reg_alpha                 0.06132587025321563
#         reg_lambda                0.2894586630104575
#         min_split_gain            0.020584494295802447
#         max_bin                   501

# [6/7] Training final model + cross-validation ...
#       Running 10-fold stratified cross-validation ...

#       CV weighted F1 per fold : [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0)]
#       Mean  : 1.0000
#       Std   : 0.0000  (lower = more stable)

#       Held-out test report:
#               precision    recall  f1-score   support

#         HIGH      1.000     1.000     1.000      1366
#          LOW      1.000     1.000     1.000      1627
#       NORMAL      1.000     1.000     1.000      1709

#     accuracy                          1.000      4702
#    macro avg      1.000     1.000     1.000      4702
# weighted avg      1.000     1.000     1.000      4702

#       Confusion matrix:
#                Pred HIGH  Pred NORMAL  Pred LOW
# Actual HIGH         1366            0         0
# Actual NORMAL          0         1709         0
# Actual LOW             0            0      1627

#       Feature importances (split gain):
#         z_score                           3136  ███████████████████████████████████
#         item_enc                          1420  ███████████████
#         item_mean                         1327  ██████████████
#         slot_count                        1261  ██████████████
#         slot_count_norm                   1103  ████████████
#         item_std                           931  ██████████
#         item_popularity_pct                462  █████
#         item_median                         91  █
#         dow_month_interaction               28  
#         item_range                          27  
#         month                                9  
#         season                               6  
#         day_of_week                          5  
#         week_of_year                         4  
#         weekend_season                       0  
#         is_christmas_period                  0  
#         is_summer_holiday                    0  
#         is_mid_month                         0  
#         is_month_end                         0  
#         is_month_start                       0  
#         is_monday                            0  
#         is_friday                            0  
#         is_weekend                           0  
#         day_of_month                         0  
#         quarter                              0  
#         month_start_weekend                  0  

# [7/7] Calibrating & saving ...
#       ✅ demand_model.pkl
#       ✅ demand_model_calibrated.pkl  ← use this in FastAPI
#       ✅ le_item.pkl
#       ✅ feature_names.pkl

# ============================================================
#   INFERENCE EXAMPLE
# ============================================================

#   Input  → Whole Milk | Saturday Dec 21 | Christmas period
#   Output → demand = HIGH  (confidence: 100%)
#   Probs  → {'HIGH': '100.0%', 'LOW': '0.0%', 'NORMAL': '0.0%'}

# ✅ Pipeline complete!

"""
Dynamic Pricing — Demand Prediction Model
Hardware target : Apple M1 Max, 32GB RAM
Model           : LightGBM + Optuna hyperparameter tuning
Expected runtime: 10–20 minutes
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import warnings
import time
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection  import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import classification_report, confusion_matrix, f1_score
from sklearn.calibration      import CalibratedClassifierCV

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
INPUT_FILE     = "Groceries_dataset.csv"
MODEL_OUT      = "demand_model.pkl"
ENCODER_OUT    = "le_item.pkl"
CALIBRATED_OUT = "demand_model_calibrated.pkl"
OPTUNA_TRIALS  = 80    # ~15–20 mins on M1 Max; increase to 150 for best results
CV_FOLDS       = 10
RANDOM_STATE   = 42
# ──────────────────────────────────────────────

print("=" * 60)
print("  DEMAND MODEL TRAINING PIPELINE")
print("=" * 60)


# ── 1. Load & parse dates
print("\n[1/7] Loading data ...")
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Date"])
df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.title()
df = df[df["itemDescription"] != "Nan"]
print(f"      {len(df):,} rows loaded")


# ── 2. Rich feature engineering
print("\n[2/7] Engineering features ...")

df["day_of_week"]    = df["Date"].dt.dayofweek
df["month"]          = df["Date"].dt.month
df["day_of_month"]   = df["Date"].dt.day
df["week_of_year"]   = df["Date"].dt.isocalendar().week.astype(int)
df["quarter"]        = df["Date"].dt.quarter
df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
df["is_friday"]      = (df["day_of_week"] == 4).astype(int)   # pre-weekend spike
df["is_monday"]      = (df["day_of_week"] == 0).astype(int)   # post-weekend drop
df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)  # payday effect
df["is_month_end"]   = (df["day_of_month"] >= 28).astype(int)
df["is_mid_month"]   = (
    (df["day_of_month"] >= 14) & (df["day_of_month"] <= 16)
).astype(int)

df["season"] = df["month"].map({
    12:0, 1:0, 2:0,   # Winter
    3:1,  4:1, 5:1,   # Spring
    6:2,  7:2, 8:2,   # Summer
    9:3, 10:3, 11:3   # Autumn
})

# Holiday proximity (no library needed)
df["is_christmas_period"] = (
    ((df["month"] == 12) & (df["day_of_month"] >= 18)) |
    ((df["month"] ==  1) & (df["day_of_month"] <=  2))
).astype(int)
df["is_summer_holiday"] = (
    (df["month"] >= 7) & (df["month"] <= 8)
).astype(int)

# Slot demand — how many purchases of this item on this day_of_week + month
df["slot_count"] = df.groupby(
    ["itemDescription", "day_of_week", "month"]
)["Date"].transform("count")

# Per-item baseline stats
item_stats = (
    df.groupby("itemDescription")["slot_count"]
      .agg(["mean","std","median","min","max"])
      .reset_index()
)
item_stats.columns = [
    "itemDescription","item_mean","item_std",
    "item_median","item_min","item_max"
]
item_stats["item_std"]   = item_stats["item_std"].fillna(1).replace(0, 1)
item_stats["item_range"] = item_stats["item_max"] - item_stats["item_min"]
df = df.merge(item_stats, on="itemDescription", how="left")

# Z-score — how unusual is this slot relative to this item's baseline
df["z_score"] = (df["slot_count"] - df["item_mean"]) / df["item_std"]

# Normalised slot count 0–1 within each item
df["slot_count_norm"] = (df["slot_count"] - df["item_min"]) / (
    df["item_range"].replace(0, 1)
)

# Interaction features
df["dow_month_interaction"] = df["day_of_week"] * 12 + df["month"]  # captures Dec Saturdays
df["weekend_season"]        = df["is_weekend"] * df["season"]
df["month_start_weekend"]   = df["is_month_start"] * df["is_weekend"]

# Item global popularity percentile rank
pop = df.groupby("itemDescription").size().rank(pct=True)
df["item_popularity_pct"] = df["itemDescription"].map(pop)

print(f"      Features engineered ✅")


# ── 3. Create demand label using z-score
print("\n[3/7] Creating demand labels ...")

def label(z):
    if   z >= 0.5:  return "HIGH"
    elif z >= -0.5: return "NORMAL"
    else:           return "LOW"

df["demand_label"] = df["z_score"].apply(label)
dist = df["demand_label"].value_counts()
for lbl in ["HIGH","NORMAL","LOW"]:
    cnt = dist.get(lbl, 0)
    print(f"      {lbl:<8} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")


# ── 4. Encode & prepare
print("\n[4/7] Encoding features ...")

le = LabelEncoder()
df["item_enc"] = le.fit_transform(df["itemDescription"])

FEATURES = [
    # Item identity
    "item_enc",

    # Core time features — all known BEFORE any sale happens
    "day_of_week", "month", "quarter", "week_of_year", "day_of_month",
    "season",

    # Binary time flags
    "is_weekend", "is_friday", "is_monday",
    "is_month_start", "is_month_end", "is_mid_month",
    "is_christmas_period", "is_summer_holiday",

    # Interaction features (time only — no sales data involved)
    "dow_month_interaction", "weekend_season", "month_start_weekend",

    # REMOVED — derived from slot_count/z_score which IS the label formula:
    # "slot_count", "slot_count_norm", "z_score",
    # "item_mean", "item_std", "item_median", "item_range", "item_popularity_pct",
]

X = df[FEATURES].copy()
y = df["demand_label"]
print(f"      {len(FEATURES)} features × {len(X):,} rows")


# ── 5. Train / val split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


# ── 6. Optuna hyperparameter search
print(f"\n[5/7] Optuna search — {OPTUNA_TRIALS} trials, grab a coffee ☕ ...")
start = time.time()

def objective(trial):
    # "goss" removed — it does not support subsample/bagging params
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "verbosity":         -1,
        "boosting_type":     boosting_type,
        "num_leaves":        trial.suggest_int("num_leaves", 64, 1024),
        "max_depth":         trial.suggest_int("max_depth", 6, 24),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin":           trial.suggest_int("max_bin", 128, 512),
        "class_weight":      "balanced",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }

    # subsample (bagging) is only valid for gbdt
    # dart uses dropout instead of bagging so these params are ignored/invalid
    if boosting_type == "gbdt":
        params["subsample"]      = trial.suggest_float("subsample", 0.5, 1.0)
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 7)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1)
        ]
    )
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="weighted")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=15)
)
# catch=(Exception,) → failed trials are skipped not crash the whole run
study.optimize(objective, n_trials=OPTUNA_TRIALS,
               show_progress_bar=True, catch=(Exception,))

elapsed = time.time() - start
print(f"\n      ✅ Tuning complete in {elapsed/60:.1f} mins")
print(f"      Best weighted F1 : {study.best_value:.4f}")
print(f"      Best params:")
for k, v in study.best_params.items():
    print(f"        {k:<25} {v}")


# ── 7. Train final model with best params
print("\n[6/7] Training final model + cross-validation ...")

best_params = {
    **study.best_params,
    "objective":    "multiclass",
    "num_class":    3,
    "metric":       "multi_logloss",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
    "verbosity":    -1,
}

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_tr, y_tr)

# 10-fold cross-validation for unbiased performance estimate
print(f"      Running {CV_FOLDS}-fold stratified cross-validation ...")
cv_model = lgb.LGBMClassifier(**best_params)
cv_scores = cross_val_score(
    cv_model, X, y,
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1_weighted",
    n_jobs=-1
)
print(f"\n      CV weighted F1 per fold : {[round(s,3) for s in cv_scores]}")
print(f"      Mean  : {cv_scores.mean():.4f}")
print(f"      Std   : {cv_scores.std():.4f}  (lower = more stable)")

# Held-out evaluation
y_pred = final_model.predict(X_val)
print(f"\n      Held-out test report:")
print(classification_report(y_val, y_pred, digits=3))

print("      Confusion matrix:")
cm = confusion_matrix(y_val, y_pred, labels=["HIGH","NORMAL","LOW"])
cm_df = pd.DataFrame(cm,
    index  =["Actual HIGH","Actual NORMAL","Actual LOW"],
    columns=["Pred HIGH",  "Pred NORMAL",  "Pred LOW"]
)
print(cm_df.to_string())

# Feature importance
print("\n      Feature importances (split gain):")
imp_series = pd.Series(
    final_model.feature_importances_, index=FEATURES
).sort_values(ascending=False)
for feat, imp in imp_series.items():
    bar = "█" * int(imp / imp_series.max() * 35)
    print(f"        {feat:<30} {imp:>7.0f}  {bar}")


# ── 8. Calibrate probabilities & save
print("\n[7/7] Calibrating & saving ...")

# Calibration makes confidence scores trustworthy for the pricing engine
# "87% confident HIGH" should actually be right ~87% of the time
calibrated = CalibratedClassifierCV(final_model, cv=5, method="isotonic")
calibrated.fit(X_tr, y_tr)

joblib.dump(final_model, MODEL_OUT)
joblib.dump(calibrated,  CALIBRATED_OUT)
joblib.dump(le,          ENCODER_OUT)
joblib.dump(FEATURES,    "feature_names.pkl")

print(f"      ✅ {MODEL_OUT}")
print(f"      ✅ {CALIBRATED_OUT}  ← use this in FastAPI")
print(f"      ✅ {ENCODER_OUT}")
print(f"      ✅ feature_names.pkl")


# ── Inference example — how FastAPI will call this
print("\n" + "=" * 60)
print("  INFERENCE EXAMPLE")
print("=" * 60)

try:
    sample = pd.DataFrame([{
        "item_enc"             : le.transform(["Whole Milk"])[0],
        "day_of_week"          : 5,      # Saturday
        "month"                : 12,     # December
        "quarter"              : 4,
        "week_of_year"         : 51,
        "day_of_month"         : 21,
        "season"               : 0,      # Winter
        "is_weekend"           : 1,
        "is_friday"            : 0,
        "is_monday"            : 0,
        "is_month_start"       : 0,
        "is_month_end"         : 0,
        "is_mid_month"         : 0,
        "is_christmas_period"  : 1,
        "is_summer_holiday"    : 0,
        "dow_month_interaction": 5 * 12 + 12,
        "weekend_season"       : 1 * 0,
        "month_start_weekend"  : 0,
    }])

    proba   = calibrated.predict_proba(sample)[0]
    classes = calibrated.classes_
    pred    = classes[np.argmax(proba)]
    conf    = float(np.max(proba))

    print(f"\n  Input  → Whole Milk | Saturday Dec 21 | Christmas period")
    print(f"  Output → demand = {pred}  (confidence: {conf:.0%})")
    print(f"  Probs  → { {c: f'{p:.1%}' for c, p in zip(classes, proba)} }")
except Exception as e:
    print(f"  (Inference example skipped: {e})")

print("\n✅ Pipeline complete!")