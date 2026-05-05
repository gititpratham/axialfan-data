from __future__ import annotations

"""
data.py — Raw test data and engineering calculations for 18" Tube Axial Fan.

Implements the full calculation sheet:
  WT, WTd, Input KW, PF, Volume (Qt), Velocities,
  VPot, VPi, TPot, TPi, FTPT, Mo, Q, FTP, FSP,
  BKW, Air Power, Efficiencies, Motor Input.
"""

import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────
# Raw 25-point test data
# ────────────────────────────────────────────────────────────────
TA18 = {
    'Srno': list(range(1, 26)),
    'ANGLE': [20]*5 + [30]*5 + [35]*5 + [40]*5 + [45]*5,
    'DEL_P': [8, 6, 3.5, 2, 0.5,
              9.5, 6.5, 5, 1.5, 0.2,
              9.5, 8, 3, 1, 0.2,
              12, 9, 3, 1, 0.2,
              14, 8.5, 3.5, 1.5, 0.5],
    'SP':    [7.5, 9.5, 11.5, 13.5, 16.5,
              9, 10, 11, 13, 15,
              9, 10.5, 11, 12.5, 15,
              11.5, 12, 10.5, 13, 15,
              12.5, 11.5, 9.5, 11, 13],
    'W1':    [45, 50, 50, 60, 50,
              48, 50, 55, 56, 55,
              45, 50, 55, 60, 55,
              60, 60, 60, 60, 60,
              55, 55, 60, 60, 55],
    'W2':    [5, 6, 6, 7, 6,
              8, 9, 10, 9, 9,
              9, 9, 10, 10, 10,
              10, 11, 12, 12, 13,
              9, 9, 10, 8, 8],
    'Volt':  [427, 431, 434, 434, 433,
              428, 431, 434, 434, 435,
              445, 445, 440, 439, 442,
              441, 441, 441, 441, 441,
              439, 440, 439, 440, 441],
    'Amp':   [1.33, 1.33, 1.33, 1.34, 1.41,
              1.323, 1.325, 1.293, 1.33, 1.39,
              1.45, 1.44, 1.36, 1.39, 1.5,
              1.48, 1.44, 1.41, 1.46, 1.55,
              1.53, 1.47, 1.42, 1.48, 1.57],
    'RPM':   [1459, 1461, 1463, 1458, 1451,
              1452, 1453, 1460, 1456, 1445,
              1457, 1460, 1454, 1451, 1442,
              1441, 1443, 1449, 1443, 1439,
              1437, 1439, 1447, 1441, 1431],
}

TA24 = {
    'Srno': list(range(1, 26)),
    'ANGLE': [20]*5 + [30]*5 + [35]*5 + [37]*5 + [40]*5,
    'DEL_P': [8, 6, 3.8, 1.5, 0.2,
              10,7,4,2,0.6,
              10, 7, 3.5, 1.5, 0.3,
              19.5,15,7.5,3,1,
              5,3.5,2,1.2,0.6],
    'SP':    [7.5, 9.5, 12, 15, 17.20,
              10,11.5,15,17,20,
              9.5, 11.5, 14, 16, 17,
              19,22,28,32,35,
              5,6,8.5,11,14],
    'W1':    [32, 35, 40, 40, 40,
              38,39,38,39,38,
              38, 39, 39, 39, 40,
              60,59,55,47,45,
              50,50,50,50,40],
    'W2':    [10, 10, 12, 13, 13,
              10,10,10,10,11,
              10, 10, 11, 10, 11,
              32,33,33,37,39,
              20,20,20,20,10],
    'Volt':  [430, 430, 430, 430, 432,
              437,436,436,437,436,
              437, 439, 438, 436, 436,
              425,424,427,423,424,
              427,430,428,427,427],
    'Amp':   [2.66, 2.69, 2.64, 2.67, 2.73,
              2.76, 2.77, 2.76, 2.78, 2.81,
              2.76, 2.77, 2.76, 2.78, 2.81,
              2.72,2.75,2.73, 2.8, 3.09,
              1.21, 1.21, 1.20, 1.22, 1.22],
    'RPM':   [980, 978, 978, 978, 974,
              976, 977, 978, 976, 977,
              976, 977, 976, 976, 977,
              1461,1459,1459,1455,1443,
              946,945,943,942,938],
}

# ────────────────────────────────────────────────────────────────
# Default test / design constants — one dict per fan size
# ────────────────────────────────────────────────────────────────
DEFAULT_CONSTANTS = {
    'duct_dia_m':        0.4572,   # 18 in → m
    'discharge_coeff':   0.98,     # CD
    'test_temp_c':       30,       # °C
    'test_baro_mmhg':    760,      # mm Hg
    'design_baro_mmhg':  760,      # mm Hg
    'design_temp_c':     30,       # °C
    'design_speed_rpm':  1460,     # RPM
    'motor_efficiency':  0.81,     # 72 %
    'cw':                10.0,     # Wattmeter correction (CT/PT ratio)
    'g':                 9.81,     # m/s²
}

DEFAULT_CONSTANTS_24 = {
    'duct_dia_m':        0.6096,   # 24 in → m
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  978,      # RPM (from test data)
    'motor_efficiency':  0.81,
    'cw':                10.0,
    'g':                 9.81,
}

# Map fan label → (raw_data_dict, default_constants)
FAN_REGISTRY = {
    '18" Tube Axial Fan': (TA18, DEFAULT_CONSTANTS),
    '24" Tube Axial Fan': (TA24, DEFAULT_CONSTANTS_24),
}


def get_raw_data(fan: str = '18" Tube Axial Fan') -> pd.DataFrame:
    """Return a fresh copy of the raw test data for the selected fan."""
    raw, _ = FAN_REGISTRY[fan]
    return pd.DataFrame(raw).copy()


def compute_derived_quantities(
    df=None,
    constants=None,
    fan: str = '18" Tube Axial Fan',
) -> pd.DataFrame:
    """
    Compute every derived engineering quantity from the raw test data
    using the supplied (or default) constants.

    Parameters
    ----------
    df        : optional raw DataFrame; if None, loaded from FAN_REGISTRY[fan]
    constants : optional dict of engineering constants; defaults to the
                constants registered for *fan*
    fan       : '18" Tube Axial Fan' or '24" Tube Axial Fan'

    Returns a DataFrame with ~30+ columns.
    """
    if df is None:
        df = get_raw_data(fan)
    if constants is None:
        _, constants = FAN_REGISTRY[fan]
        constants = constants.copy()


    df = df.copy()

    # ── unpack constants ───────────────────────────────────────
    D        = constants['duct_dia_m']
    CD       = constants['discharge_coeff']
    Ts       = constants['test_temp_c']
    B        = constants['test_baro_mmhg']
    B_d      = constants['design_baro_mmhg']
    T_d      = constants['design_temp_c']
    N        = constants['design_speed_rpm']
    eta_m    = constants['motor_efficiency']
    CW       = constants['cw']
    g        = constants['g']

    A  = np.pi / 4 * D**2          # outlet area  (m²)
    Ai = A                          # inlet area   (m²)

    # 1  Test air density  (kg/m³)
    df['WT'] = (1.205
                * (B + 0.0737 * df['SP']) / 760
                * 293 / (273 + Ts))

    # 2  Design air density (kg/m³)
    WTd = 1.205 * B_d / 760 * 293 / (273 + T_d)
    df['WTd'] = WTd

    # 3  Input power  (kW)
    df['Mi_kW'] = (df['W1'] + df['W2']) * CW

    # 4  Power factor  (3-phase, two-wattmeter)
    df['PF'] = (df['Mi_kW'] * 1000
                / (np.sqrt(3) * df['Volt'] * df['Amp']))

    # 5  Volume flow at test conditions  (m³/hr)
    df['Qt_CMH'] = 12500 * CD * D**2 * np.sqrt(df['DEL_P'] / df['WT'])

    # 6  Outlet velocity  (m/s  &  m/hr)
    df['V_out_mps'] = df['Qt_CMH'] / (A * 3600)
    df['V_out_mhr'] = df['Qt_CMH'] / A

    # 7  Inlet velocity  (m/hr)
    df['V_in_mhr'] = df['Qt_CMH'] / Ai

    # 8  Velocity pressure – outlet  (mm WG)
    df['VPot'] = (df['V_out_mps']**2 / (2 * g)) * df['WT']

    # 9  Velocity pressure – inlet  (mm WG)
    df['VPi'] = (df['V_in_mhr'] / 16000)**2 * df['WT']

    # 10  Total pressure – outlet  (mm WG)
    df['TPot'] = df['SP'] + df['VPot']

    # 11  Total pressure – inlet  (mm WG)  (free inlet → SPi = 0)
    df['SPi'] = 0.0
    df['TPi'] = df['VPi'] - df['SPi']

    # 12  Fan total pressure – test  (mm WG)
    df['FTPT'] = df['TPot'] - df['TPi']

    # 13  Motor output  (kW)
    df['Mo_kW'] = df['Mi_kW'] * eta_m

    # 14  Rated volume  (m³/hr)  — fan-law speed correction
    df['Q_CMH'] = df['Qt_CMH'] * (N / df['RPM'])

    # 15  Rated outlet velocity  (m/hr)
    df['Rated_V_out_mhr'] = df['Q_CMH'] / A

    # 16  Rated velocity pressure – outlet  (mm WG)
    df['R_VPo'] = (df['Rated_V_out_mhr'] / 16000)**2 * WTd

    # 17  Fan total pressure – rated  (mm WG)
    df['FTP'] = df['FTPT'] * (N / df['RPM'])**2 * (WTd / df['WT'])

    # 18  Fan static pressure  (mm WG)
    df['FSP'] = df['FTP'] - df['R_VPo']

    # 19  Brake kW
    df['BKW'] = df['Mo_kW'] * (N / df['RPM'])**3 * (WTd / df['WT'])

    # 20  Air power – static  (kW)
    df['Air_Power_ST'] = 2.723 * df['Q_CMH'] * df['FSP'] * 1e-6

    # 21  Air power – total  (kW)
    df['Air_Power_T'] = 2.723 * df['Q_CMH'] * df['FTP'] * 1e-6

    # 22  Static efficiency  (%)
    df['Static_Eff'] = np.where(
        df['BKW'] > 0,
        df['Air_Power_ST'] / df['BKW'] * 100,
        0,
    )

    # 23  Total efficiency  (%)
    df['Total_Eff'] = np.where(
        df['BKW'] > 0,
        df['Air_Power_T'] / df['BKW'] * 100,
        0,
    )

    # 24  Motor input at rated conditions  (kW)
    df['Motor_Input'] = df['Mi_kW'] * (N / df['RPM'])**3 * (WTd / df['WT'])

    # store metadata for downstream use
    df.attrs['constants']   = constants
    df.attrs['outlet_area'] = A
    df.attrs['inlet_area']  = Ai

    return df
