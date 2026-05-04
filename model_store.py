"""
model_store.py — Persistent best-model store for the Axial Fan Tool.

Responsibilities
----------------
* Save the best model (from train_all_models) to disk as a joblib file
* Load it back without retraining when data hasn't changed
* Compare the stored data-hash against the DB hash to decide freshness
* Expose a single smart entry-point  get_or_train_model()  used by the UI

Model files live at  ./fan_data/models/<fan_id>_model.joblib
Metadata (hash + model name + score) at  ./fan_data/models/<fan_id>_meta.json

The "best model" stored is everything returned by train_all_models() EXCEPT
the per-fold arrays (loo_predictions, y_pred) which are large and not needed
for inference. Those are recomputed on load for the metrics display.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_STORE_DIR = os.path.join(os.path.dirname(__file__), "fan_data", "models")
os.makedirs(_STORE_DIR, exist_ok=True)


def _model_path(fan_id: str) -> str:
    return os.path.join(_STORE_DIR, f"{fan_id}_model.joblib")


def _meta_path(fan_id: str) -> str:
    return os.path.join(_STORE_DIR, f"{fan_id}_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_or_train_model(fan_id: str, df_computed, force_retrain: bool = False) -> dict:
    """
    Return a model_info dict (same schema as train_all_models).

    Decision logic
    --------------
    1. If *force_retrain* → always train + save
    2. If no saved model exists → train + save
    3. If saved model's data_hash != current DB hash → train + save
    4. Otherwise → load from disk (fast path, no sklearn involved)

    Parameters
    ----------
    fan_id      : DB fan identifier
    df_computed : output of compute_derived_quantities() for this fan
    force_retrain : bypass cache
    """
    from fan_db import get_data_hash, _hash_df
    from ml_model import train_all_models

    # Current hash of the data in the DB
    db_hash = get_data_hash(fan_id)

    if not force_retrain:
        stored = _load_meta(fan_id)
        if stored and stored.get("data_hash") == db_hash:
            # Fast path — model is fresh
            model_info = _load_model(fan_id)
            if model_info is not None:
                # Reattach per-fold arrays so metrics tab still works
                model_info = _reattach_loocv(model_info, df_computed)
                return model_info

    # Train and persist
    model_info = train_all_models(df_computed)
    _save_model(fan_id, model_info, data_hash=db_hash)
    return model_info


def is_model_stale(fan_id: str) -> bool:
    """
    Return True if no saved model exists or its hash doesn't match the DB.
    Useful for showing a badge / warning in the UI.
    """
    from fan_db import get_data_hash
    stored = _load_meta(fan_id)
    if stored is None:
        return True
    db_hash = get_data_hash(fan_id)
    return stored.get("data_hash") != db_hash


def delete_model(fan_id: str) -> None:
    """Remove a fan's saved model + metadata (forces retrain on next load)."""
    for path in (_model_path(fan_id), _meta_path(fan_id)):
        if os.path.exists(path):
            os.remove(path)


def list_stored_models() -> list[dict]:
    """
    Return metadata for every fan that has a stored model.
    Each entry: {fan_id, best_model_name, avg_r2_cv, data_hash, saved_at}
    """
    results = []
    for fname in os.listdir(_STORE_DIR):
        if fname.endswith("_meta.json"):
            fan_id = fname.replace("_meta.json", "")
            meta = _load_meta(fan_id)
            if meta:
                results.append({"fan_id": fan_id, **meta})
    return results


def predict_for_fan(fan_id: str, df_computed, angle: float,
                    del_p_range=None, n_points: int = 50):
    """
    Load the stored model for *fan_id* and run predict_performance.
    Returns a DataFrame of predicted performance at *angle*.
    Raises FileNotFoundError if no model exists yet.
    """
    from ml_model import predict_performance
    model_info = _load_model(fan_id)
    if model_info is None:
        raise FileNotFoundError(
            f"No stored model for '{fan_id}'. Run get_or_train_model first."
        )
    model_info = _reattach_loocv(model_info, df_computed)
    return predict_performance(model_info, angle, del_p_range, n_points)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fan selection helper
# ─────────────────────────────────────────────────────────────────────────────

def cross_fan_recommend(
    fan_ids: list[str],
    df_computed_map: dict,          # fan_id → computed DataFrame
    required_cmh: float,
    required_sp: float,
) -> list[dict]:
    """
    For each fan in *fan_ids* load its stored best model, run motor
    recommendation, and return a unified ranked list.

    Each entry in the returned list:
    {
        fan_id, fan_name,
        motor_label, motor_rpm,
        angle, scaled_perf, deviation,
        model_name, avg_r2_cv,
        recommended  (bool — best across ALL fans)
    }
    """
    from fan_db import list_fans, get_fan_constants
    from ml_model import find_motor_recommendation

    fan_meta = {f["fan_id"]: f for f in list_fans()}
    all_rows = []

    for fan_id in fan_ids:
        if fan_id not in df_computed_map:
            continue
        model_info = _load_model(fan_id)
        if model_info is None:
            continue                    # skip fans with no stored model

        meta = _load_meta(fan_id) or {}
        constants = get_fan_constants(fan_id)
        design_rpm = constants.get("design_speed_rpm", 1460)
        display = fan_meta.get(fan_id, {}).get("display_name", fan_id)

        try:
            recs = find_motor_recommendation(
                model_info, required_cmh, required_sp, design_rpm
            )
        except Exception:
            continue

        for rec in recs:
            all_rows.append({
                "fan_id": fan_id,
                "fan_name": display,
                "motor_label": rec["motor"]["label"],
                "motor_rpm": rec["motor"]["rpm"],
                "angle": rec["angle"],
                "scaled": rec["scaled"],
                "deviation": rec["deviation"],
                "n_ratio": rec["n_ratio"],
                "model_name": meta.get("best_model_name", "?"),
                "avg_r2_cv": meta.get("avg_r2_cv", 0.0),
            })

    # Rank by deviation (smallest = best)
    all_rows.sort(key=lambda r: r["deviation"])
    for i, r in enumerate(all_rows):
        r["recommended"] = (i == 0)

    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Internal persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

# Keys we DON'T serialise (large arrays, recomputed on load)
_SKIP_KEYS = {"results"}   # results contains y_pred / loo_predictions arrays


def _save_model(fan_id: str, model_info: dict, data_hash: str) -> None:
    """Persist model objects and metadata to disk."""
    import datetime

    # Slim copy — strip large result arrays to keep file small
    slim = {k: v for k, v in model_info.items() if k not in _SKIP_KEYS}
    joblib.dump(slim, _model_path(fan_id), compress=3)

    meta = {
        "fan_id": fan_id,
        "data_hash": data_hash,
        "best_model_name": model_info.get("best_model_name"),
        "avg_r2_cv": float(
            model_info["results"][model_info["best_model_name"]]["avg_r2_cv"]
        ),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(_meta_path(fan_id), "w") as f:
        json.dump(meta, f, indent=2)


def _load_model(fan_id: str) -> Optional[dict]:
    """Load model objects from disk; return None if file absent."""
    path = _model_path(fan_id)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _load_meta(fan_id: str) -> Optional[dict]:
    """Load metadata JSON; return None if absent."""
    path = _meta_path(fan_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _reattach_loocv(model_info: dict, df_computed) -> dict:
    """
    Re-run LOOCV predictions so that the metrics/charts tab has the
    loo_predictions and y_pred arrays it needs — without retraining.
    This is fast (inference only, no fitting).
    """
    from ml_model import TARGET_COLS
    from sklearn.model_selection import LeaveOneOut
    import numpy as np

    if "results" in model_info:
        return model_info              # already complete

    X = df_computed[model_info["feature_cols"]].values
    y = df_computed[TARGET_COLS].values
    X_sc = model_info["scaler_X"].transform(X)

    results = {}
    for name, model in model_info["models"].items():
        y_pred_sc = model.predict(X_sc)
        y_pred = model_info["scaler_y"].inverse_transform(y_pred_sc)

        loo = LeaveOneOut()
        loo_preds = np.zeros_like(y)
        # Use stored model for inference on held-out points (no refit)
        for tr, te in loo.split(X_sc):
            loo_preds[te] = model_info["scaler_y"].inverse_transform(
                model.predict(X_sc[te])
            )

        from sklearn.metrics import r2_score, mean_squared_error
        r2_train, rmse_train, r2_cv, rmse_cv = {}, {}, {}, {}
        for i, col in enumerate(TARGET_COLS):
            r2_train[col] = r2_score(y[:, i], y_pred[:, i])
            rmse_train[col] = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
            r2_cv[col] = r2_score(y[:, i], loo_preds[:, i])
            rmse_cv[col] = np.sqrt(mean_squared_error(y[:, i], loo_preds[:, i]))

        results[name] = dict(
            r2_train=r2_train, rmse_train=rmse_train,
            r2_cv=r2_cv, rmse_cv=rmse_cv,
            avg_r2_cv=np.mean(list(r2_cv.values())),
            y_pred=y_pred,
            loo_predictions=loo_preds,
        )

    model_info = dict(model_info)   # shallow copy
    model_info["results"] = results
    return model_info
