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

TA16 = {
    'Srno': list(range(1, 11)),
    'ANGLE': [25]*5 + [20]*5,
    'DEL_P': [6, 5, 3.5, 2, 0.5,
              3, 2.5, 2, 0.5, 0],
    'SP':    [6, 8, 10.5, 12.5, 13,
              3, 5, 9.5, 11.5, 17.5],
    'W1':    [15, 15, 15, 15, 15,
              58, 59, 59, 60, 61],
    'W2':    [10, 10, 10, 10, 10,
              12, 15, 12, 11, 10],
    'Volt':  [420, 421, 421, 422, 421,
              425, 425, 425, 425, 425],
    'Amp':   [0.83, 0.83, 0.83, 0.84, 0.82,
              0.78, 0.78, 0.79, 0.79, 0.79],
    'RPM':   [1447, 1447, 1448, 1449, 1447,
              1484, 1481, 1478, 1479, 1474],
}

TA41 = {
    'Srno': list(range(1, 56)),
    'ANGLE': [45]*5 + [25]*5 + [35]*5 + [25]*5 + [25]*5 + [22]*5 + [28]*5 + [14]*5 + [20]*5 + [25]*5 + [23]*5,
    'DEL_P': [28, 17, 8, 2, 0.2,
              8, 5.5, 4.5, 2, 0.5,
              14, 10.5, 6, 2, 0.5,
              22, 13, 6, 2, 1,
              14, 10, 6, 1, 0.5,
              14, 10, 6, 2, 0.5,
              22, 13, 6, 2, 0.5,
              18.5, 12, 5, 2, 0.5,
              11.5, 7.5, 5, 2.5, 0.5,
              13.5, 8.5, 4.6, 2, 0.5,
              11.7, 7.5, 4.5, 3, 1],
    'SP':    [28, 33.5, 41, 43, 33,
              8, 12, 21, 29.5, 34,
              14, 18.5, 29.5, 37, 36,
              22, 27.5, 37, 42, 43,
              14, 18, 20, 22, 23,
              14, 19, 29.5, 36, 37,
              22, 28, 37, 42, 43,
              18.5, 27.5, 30, 37.5, 33,
              11.5, 20.5, 22.5, 34.5, 37,
              13.5, 23, 29, 36.5, 35,
              12.5, 20.5, 28.5, 34, 36],
    'W1':    [100, 100, 100, 100, 90,
              40, 40, 42, 40, 40,
              40, 40, 40, 40, 35,
              60, 60, 60, 65, 65,
              60, 60, 60, 65, 60,
              40, 40, 40, 40, 40,
              60, 60, 60, 65, 65,
              95, 95, 95, 95, 100,
              100, 100, 101, 101, 101,
              85, 85, 88, 85, 89,
              75, 75, 75, 75, 75],
    'W2':    [40, 40, 40, 40, 30,
              20, 20, 20, 20, 18,
              20, 20, 20, 15, 10,
              40, 45, 45, 40, 40,
              30, 30, 30, 30, 20,
              20, 20, 20, 20, 13,
              40, 45, 45, 40, 40,
              100, 95, 95, 100, 110,
              102, 102, 105, 103, 103,
              100, 102, 105, 102, 109,
              85, 89, 89, 90, 90],
    'Volt':  [437, 438, 437, 436, 438,
              426, 426, 427, 426, 426,
              425, 426, 427, 426, 426,
              431, 430, 430, 431, 431,
              434, 433, 434, 433, 435,
              438, 438, 435, 435, 432,
              431, 430, 431, 431, 430,
              410, 410, 410, 410, 410,
              440, 440, 440, 440, 440,
              435, 435, 435, 435, 435,
              410, 410, 410, 410, 410],
    'Amp':   [17.05, 17.02, 17.1, 18.5, 19.19,
              5, 5.05, 5.16, 5.26, 5.27,
              5.63, 5.84, 5.97, 6.22, 6.63,
              10.5, 10.8, 11.05, 11.8, 11.2,
              10.11, 10.15, 10.7, 10.8, 10.04,
              5.69, 5.82, 6, 6.22, 6.48,
              10.5, 10.8, 11.05, 11.8, 11.2,
              7.3, 7.66, 7.34, 8.33, 9.48,
              5.35, 5.52, 5.6, 5.85, 5.93,
              5.95, 6.1, 6.15, 6.48, 6.95,
              5.21, 5.36, 5.5, 5.68, 5.82],
    'RPM':   [983, 986, 983, 981, 977,
              989, 989, 989, 987, 986,
              983, 980, 982, 978, 975,
              978, 977, 979, 976, 972,
              986, 983, 981, 979, 975,
              986, 983, 981, 979, 975,
              978, 977, 979, 976, 972,
              974.9, 972, 972.7, 967.7, 959,
              991, 989.1, 987.5, 986.6, 989.2,
              981.3, 981, 980, 978.6, 978,
              987.2, 985.8, 985.4, 983.5, 984.4],
}

TA48 = {
    'Srno': list(range(1, 16)),
    'ANGLE': [30]*5 + [24]*5 + [28]*5,
    'DEL_P': [22, 19, 16, 8.5, 1,
              15, 13.5, 8.5, 3, 0.5,
              18, 15, 8, 3, 0.5],
    'SP':    [21, 31, 58, 98, 112,
              14, 22, 40.5, 62, 75,
              18, 26, 45, 71, 85],
    'W1':    [120, 120, 135, 140, 100,
              40, 40, 40, 40, 40,
              115, 120, 125, 135, 130],
    'W2':    [150, 155, 165, 175, 120,
              20, 20, 20, 12, 10,
              0, 5, 10, 20, 10],
    'Volt':  [424, 425, 426, 429, 427,
              433, 434, 434, 431, 434,
              425, 425, 425, 425, 425],
    'Amp':   [17.3, 17.9, 18.8, 19.8, 15.2,
              8.47, 8.67, 9.16, 10.15, 9.53,
              13.5, 13.9, 14.1, 14.6, 14.3],
    'RPM':   [737, 737, 735, 734, 740,
              1480, 1479, 1473, 1466, 1470,
              1490, 1486, 1487, 1485, 1483],
}

TA54 = {
    'Srno': list(range(1, 16)),
    'ANGLE': [10]*5 + [24]*5 + [28]*5,
    'DEL_P': [30, 25, 16, 7, 3,
              28.5, 23.5, 10.5, 2.5, 0.5,
              30, 35.5, 17, 4.5, 0.5],
    'SP':    [30, 50, 88, 130, 142,
              35.5, 50.5, 69.5, 73.5, 78.5,
              51, 55.5, 75, 79.5, 84.5],
    'W1':    [140, 145, 140, 135, 135,
              100, 100, 100, 100, 100,
              100, 110, 110, 110, 110],
    'W2':    [110, 110, 110, 105, 105,
              60, 60, 70, 65, 60,
              70, 85, 85, 70, 70],
    'Volt':  [435, 435, 435, 435, 435,
              435, 435, 435, 435, 435,
              430, 430, 430, 430, 430],
    'Amp':   [24, 24.3, 24.3, 23.3, 22,
              20.4, 21, 21.2, 21.1, 20.5,
              22.7, 23.7, 23.6, 23.2, 23.3],
    'RPM':   [1485, 1484, 1484, 1484, 1483,
              993.6, 992.6, 991.9, 993.3, 995.7,
              992, 992.4, 991.4, 990.9, 989.8],
}

# ────────────────────────────────────────────────────────────────
# Default test / design constants — one dict per fan size
# ────────────────────────────────────────────────────────────────
DEFAULT_CONSTANTS_14 = {
    'duct_dia_m':        0.3556,   # 14 in → m (avg of 12" and 16")
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  2155,     # RPM (avg of 12" [2850] and 16" [1460])
    'motor_efficiency':  0.85,     # avg of 12" [0.89] and 16" [0.81]
    'cw':                4.0,
    'g':                 9.81,
}

DEFAULT_CONSTANTS_16 = {
    'duct_dia_m':        0.4064,   # 16 in → m
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  1460,     # RPM
    'motor_efficiency':  0.81,
    'cw':                4.0,
    'g':                 9.81,
}

DEFAULT_CONSTANTS = {
    'duct_dia_m':        0.4572,   # 18 in → m
    'discharge_coeff':   0.98,     # CD
    'test_temp_c':       30,       # °C
    'test_baro_mmhg':    760,      # mm Hg
    'design_baro_mmhg':  760,      # mm Hg
    'design_temp_c':     30,       # °C
    'design_speed_rpm':  1460,     # RPM
    'motor_efficiency':  0.81,     # 72 %
    'cw':                6.6,      # Wattmeter correction (CT/PT ratio)
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
    'cw':                12.7,
    'g':                 9.81,
}

DEFAULT_CONSTANTS_41 = {
    'duct_dia_m':        1.0414,   # 41 in → m
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  980,      # RPM
    'motor_efficiency':  0.81,
    'cw':                20.0,
    'g':                 9.81,
}

DEFAULT_CONSTANTS_48 = {
    'duct_dia_m':        1.2192,   # 48 in → m
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  1460,     # RPM
    'motor_efficiency':  0.81,
    'cw':                80.0,
    'g':                 9.81,
}

DEFAULT_CONSTANTS_54 = {
    'duct_dia_m':        1.3716,   # 54 in → m
    'discharge_coeff':   0.98,
    'test_temp_c':       30,
    'test_baro_mmhg':    760,
    'design_baro_mmhg':  760,
    'design_temp_c':     30,
    'design_speed_rpm':  980,      # RPM
    'motor_efficiency':  0.81,
    'cw':                20.0,
    'g':                 9.81,
}

TA14 = {
    'Srno': list(range(1, 11)),
    'ANGLE': [25.0]*5 + [30.0]*5,
    'DEL_P': [4.75, 3.75, 2.25, 0.85, 0.10, 7.50, 5.25, 3.00, 1.60, 0.35],
    'SP':    [4.75, 7.00, 15.50, 21.25, 28.75, 9.00, 15.00, 21.00, 23.50, 27.00],
    'W1':    [34.0, 34.5, 34.5, 35.0, 35.5, 18.5, 18.0, 17.5, 17.5, 17.5],
    'W2':    [12.0, 13.5, 12.0, 11.5, 11.0, 15.0, 15.0, 15.5, 15.5, 15.5],
    'Volt':  [430.5, 430.0, 429.0, 429.5, 428.5, 429.0, 428.0, 429.0, 429.0, 429.5],
    'Amp':   [0.80, 0.80, 0.80, 0.82, 0.83, 1.08, 1.10, 1.18, 1.23, 1.34],
    'RPM':   [2209.0, 2208.5, 2203.5, 2191.5, 2175.5, 2156.5, 2138.5, 2111.5, 2092.0, 2029.0],
}

# Map fan label → (raw_data_dict, default_constants)
FAN_REGISTRY = {
    '14" Tube Axial Fan': (TA14, DEFAULT_CONSTANTS_14),
    '16" Tube Axial Fan': (TA16, DEFAULT_CONSTANTS_16),
    '18" Tube Axial Fan': (TA18, DEFAULT_CONSTANTS),
    '24" Tube Axial Fan': (TA24, DEFAULT_CONSTANTS_24),
    '41" Tube Axial Fan': (TA41, DEFAULT_CONSTANTS_41),
    '48" Tube Axial Fan': (TA48, DEFAULT_CONSTANTS_48),
    '54" Tube Axial Fan': (TA54, DEFAULT_CONSTANTS_54),
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
    df['Mi_kW'] = (df['W1'] + df['W2']) * CW / 1000

    # 4  Power factor  (3-phase, two-wattmeter)
    df['PF'] = (df['Mi_kW'] * 1000
                / (np.sqrt(3) * df['Volt'] * df['Amp']))

    # 5  Volume flow at test conditions  (m³/hr)
    df['Qt_CMH'] = 12500 * CD * D**2 * np.sqrt(df['DEL_P'] / df['WT'])

    # 6  Outlet velocity  (m/s & m/hr)
    df['V_out_mps'] = df['Qt_CMH'] / (A * 3600)
    df['V_out_mhr'] = df['Qt_CMH'] / A

    # 7  Inlet velocity  (m/s  & m/hr)
    #    Ai = A for a tube-axial fan (same duct area inlet & outlet)
    df['V_in_mhr'] = df['Qt_CMH'] / Ai
    df['V_in_mps'] = df['Qt_CMH'] / (Ai * 3600)  # exact, used for VP

    # 8  Velocity pressure – outlet  (mm WG)
    #    VP (Pa) = WT × V² / 2;  1 mm WG = 9.81 Pa  →  VP (mm WG) = WT × V² / (2g)
    df['VPot'] = (df['V_out_mps']**2 / (2 * g)) * df['WT']

    # 9  Velocity pressure – inlet  (mm WG)
    #    Use the SAME exact formula as VPot (consistent units, same area).
    #    Avoids the ~0.34 % error from the rounded constant 16 000 ≈ √(2g)×3600 = 15 946.
    df['VPi'] = (df['V_in_mps']**2 / (2 * g)) * df['WT']

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

    # 15  Rated outlet velocity  (m/hr  &  m/s)
    df['Rated_V_out_mhr'] = df['Q_CMH'] / A
    df['Rated_V_out_mps'] = df['Q_CMH'] / (A * 3600)  # exact, used for VP

    # 16  Rated velocity pressure – outlet  (mm WG)
    #    Same exact formula as VPot / VPi — no 16 000 approximation.
    df['R_VPo'] = (df['Rated_V_out_mps']**2 / (2 * g)) * WTd

    # 17  Fan total pressure – rated  (mm WG)
    df['FTP'] = df['FTPT'] * (N / df['RPM'])**2 * (WTd / df['WT'])

    # 18  Fan static pressure  (mm WG)
    df['FSP'] = df['FTP'] - df['R_VPo']

    # 19  Brake kW
    df['BKW'] = df['Mo_kW'] * (N / df['RPM'])**3 * (WTd / df['WT'])

    # 20  Air power – static  (kW)
    #    Exact constant: 9.81 Pa/mm WG ÷ (1000 W/kW × 3600 s/hr) = 2.725 × 10⁻⁶
    #    FSP is clamped to 0 for static air power: when the fan operates near free
    #    delivery, FSP = FTP − R_VPo can be slightly negative (physical — all energy
    #    goes into kinetic pressure).  Negative static efficiency is meaningless.
    df['FSP_eff'] = df['FSP'].clip(lower=0)   # floor for efficiency calc only
    df['Air_Power_ST'] = 2.725 * df['Q_CMH'] * df['FSP_eff'] * 1e-6

    # 21  Air power – total  (kW)
    df['Air_Power_T'] = 2.725 * df['Q_CMH'] * df['FTP'] * 1e-6

    # 22  Static efficiency  (%)
    df['Static_Eff'] = np.where(
        df['BKW'] > 0,
        (df['Air_Power_ST'] / df['BKW']) * 100,
        0,
    )

    # 23  Total efficiency  (%)
    df['Total_Eff'] = np.where(
        df['BKW'] > 0,
        (df['Air_Power_T'] / df['BKW']) * 100,
        0,
    )

    # 24  Motor input at rated conditions  (kW)
    df['Motor_Input'] = df['Mi_kW'] * (N / df['RPM'])**3 * (WTd / df['WT'])

    # store metadata for downstream use
    df.attrs['constants']   = constants
    df.attrs['outlet_area'] = A
    df.attrs['inlet_area']  = Ai

    return df
