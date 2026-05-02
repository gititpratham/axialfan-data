from __future__ import annotations

"""
ml_model.py — Machine-learning models for fan performance prediction.

Trains Gradient Boosting, Random Forest, and Gaussian Process regressors
on the computed fan-test data (25 rows, 5 blade angles).

Features : [ANGLE, DEL_P]
Targets  : [SP, Qt_CMH, Q_CMH, FSP, FTP, BKW, Static_Eff, Total_Eff]

Uses Leave-One-Out Cross-Validation for honest evaluation on small data.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

# ────────────────────────────────────────────────────────────────
# Column definitions
# ────────────────────────────────────────────────────────────────
FEATURE_COLS = ['ANGLE', 'DEL_P']
TARGET_COLS  = ['SP', 'Qt_CMH', 'Q_CMH', 'FSP', 'FTP',
                'BKW', 'Static_Eff', 'Total_Eff']


# ────────────────────────────────────────────────────────────────
# Model factories (so we can re-create identical models per CV fold)
# ────────────────────────────────────────────────────────────────
def _make_gbr():
    return MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_split=3, min_samples_leaf=2, random_state=42,
        )
    )

def _make_rf():
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200, max_depth=4,
            min_samples_split=3, min_samples_leaf=2, random_state=42,
        )
    )

def _make_gpr():
    kernel = (ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
              + WhiteKernel(noise_level=0.1))
    return MultiOutputRegressor(
        GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=42,
        )
    )

MODEL_FACTORIES = {
    'Gradient Boosting': _make_gbr,
    'Random Forest':     _make_rf,
    'Gaussian Process':  _make_gpr,
}


# ────────────────────────────────────────────────────────────────
# Training & evaluation
# ────────────────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train every model variant, evaluate with LOOCV, and pick the best.

    Returns a dict with keys:
        models, results, best_model_name, best_model,
        scaler_X, scaler_y, feature_cols, target_cols
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    X_sc = scaler_X.transform(X)
    y_sc = scaler_y.transform(y)

    trained_models = {}
    results = {}

    for name, factory in MODEL_FACTORIES.items():
        # ── train on all data ──────────────────────────────────
        model = factory()
        model.fit(X_sc, y_sc)
        trained_models[name] = model

        y_pred = scaler_y.inverse_transform(model.predict(X_sc))

        # ── LOOCV ─────────────────────────────────────────────
        loo = LeaveOneOut()
        loo_preds = np.zeros_like(y)

        for tr, te in loo.split(X_sc):
            m = factory()
            m.fit(X_sc[tr], y_sc[tr])
            loo_preds[te] = scaler_y.inverse_transform(m.predict(X_sc[te]))

        # ── per-target metrics ─────────────────────────────────
        r2_train, rmse_train = {}, {}
        r2_cv,    rmse_cv    = {}, {}

        for i, col in enumerate(TARGET_COLS):
            r2_train[col]   = r2_score(y[:, i], y_pred[:, i])
            rmse_train[col] = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
            r2_cv[col]      = r2_score(y[:, i], loo_preds[:, i])
            rmse_cv[col]    = np.sqrt(mean_squared_error(y[:, i], loo_preds[:, i]))

        results[name] = dict(
            r2_train=r2_train, rmse_train=rmse_train,
            r2_cv=r2_cv,       rmse_cv=rmse_cv,
            avg_r2_cv=np.mean(list(r2_cv.values())),
            y_pred=y_pred,
            loo_predictions=loo_preds,
        )

    best_name = max(results, key=lambda k: results[k]['avg_r2_cv'])

    return dict(
        models=trained_models,
        results=results,
        best_model_name=best_name,
        best_model=trained_models[best_name],
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
    )


# ────────────────────────────────────────────────────────────────
# Prediction helpers
# ────────────────────────────────────────────────────────────────
def predict_performance(
    model_info: dict,
    angle: float,
    del_p_range=None,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Predict full fan performance at an arbitrary blade angle.

    Sweeps DEL_P (velocity head) from 0.1 to 16 and predicts all
    target quantities using the best trained model.
    """
    if del_p_range is None:
        del_p_range = np.linspace(0.1, 16, n_points)

    X_new = np.column_stack([
        np.full(len(del_p_range), angle),
        del_p_range,
    ])

    X_sc = model_info['scaler_X'].transform(X_new)
    y_sc = model_info['best_model'].predict(X_sc)
    y    = model_info['scaler_y'].inverse_transform(y_sc)

    out = pd.DataFrame(y, columns=model_info['target_cols'])
    out['ANGLE'] = angle
    out['DEL_P'] = del_p_range
    return out


def find_best_operating_point(
    model_info: dict,
    required_cmh: float,
    required_sp: float,
) -> tuple:
    """
    Search over angles 18°–48° to find the blade setting that best
    delivers the requested volume & static pressure.

    Returns (best_angle, best_row, min_distance).
    """
    angles = np.linspace(18, 48, 100)
    best_result = None
    best_angle  = None
    min_dist    = float('inf')

    for angle in angles:
        pred = predict_performance(model_info, angle, n_points=80)
        for _, row in pred.iterrows():
            dq = (row['Q_CMH'] - required_cmh) / max(required_cmh, 1)
            ds = ((row['FSP'] - required_sp) / max(abs(required_sp), 0.1))
            d  = np.sqrt(dq**2 + ds**2)
            if d < min_dist:
                min_dist    = d
                best_angle  = angle
                best_result = row

    return best_angle, best_result, min_dist


# ────────────────────────────────────────────────────────────────
# Standard motor catalogue & motor recommendation
# ────────────────────────────────────────────────────────────────
STANDARD_MOTORS = [
    {'rpm': 950,  'poles': 6, 'label': '950 RPM — 6 Pole'},
    {'rpm': 1440, 'poles': 4, 'label': '1440 RPM — 4 Pole'},
    {'rpm': 2850, 'poles': 2, 'label': '2850 RPM — 2 Pole'},
]


def find_motor_recommendation(
    model_info: dict,
    required_cmh: float,
    required_sp: float,
    design_rpm: float,
) -> list:
    """
    For each standard motor speed use fan laws to scale the ML-predicted
    performance and find which motor + blade-angle combination best
    delivers the required CMH and static pressure.

    Fan laws applied:
        Q2 = Q1 × (N2/N1)
        SP2 = SP1 × (N2/N1)²
        P2  = P1  × (N2/N1)³
        η   = constant (efficiency is speed-independent)

    Parameters
    ----------
    model_info   : dict returned by train_all_models()
    required_cmh : target volume flow rate (m³/hr)
    required_sp  : target static pressure (mm WG)
    design_rpm   : RPM at which the ML model was trained

    Returns
    -------
    List of dicts sorted by deviation (best first), each containing:
        motor, angle, scaled_perf, deviation, recommended
    """
    rows = []

    for motor in STANDARD_MOTORS:
        n_ratio = motor['rpm'] / design_rpm

        # Scale requirements back to the design-RPM space so the ML model
        # can work in its training domain
        cmh_target_design = required_cmh / n_ratio
        sp_target_design  = required_sp  / (n_ratio ** 2)

        # Find best angle in design-RPM space
        angle, best_row, _ = find_best_operating_point(
            model_info, cmh_target_design, sp_target_design
        )

        if best_row is None:
            continue

        # Scale back to actual motor RPM using fan laws
        scaled = {
            'Q_CMH':      best_row['Q_CMH']  * n_ratio,
            'FSP':        best_row['FSP']     * (n_ratio ** 2),
            'FTP':        best_row['FTP']     * (n_ratio ** 2),
            'BKW':        best_row['BKW']     * (n_ratio ** 3),
            'Static_Eff': best_row['Static_Eff'],   # η is speed-independent
            'Total_Eff':  best_row['Total_Eff'],
            'SP':         best_row['SP']      * (n_ratio ** 2),
        }

        # Deviation from target at actual motor RPM
        dq  = abs(scaled['Q_CMH'] - required_cmh) / max(required_cmh, 1)
        ds  = abs(scaled['FSP']   - required_sp)  / max(abs(required_sp), 0.1)
        dev = np.sqrt(dq ** 2 + ds ** 2)

        rows.append({
            'motor':      motor,
            'angle':      round(angle, 1),
            'scaled':     scaled,
            'deviation':  dev,
            'n_ratio':    n_ratio,
        })

    # Best match first
    rows.sort(key=lambda x: x['deviation'])

    # Tag the recommended one
    for i, r in enumerate(rows):
        r['recommended'] = (i == 0)

    return rows
