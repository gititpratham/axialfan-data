"""
app.py — Tube Axial Fan Performance Analysis Tool (18" & 24")

A Streamlit dashboard with four tabs:
  1. Raw Data & Calculations
  2. Performance Curves
  3. ML Predictions
  4. Fan Selection
"""

import streamlit as st
import pandas as pd
import numpy as np

from data import (
    get_raw_data, compute_derived_quantities,
    DEFAULT_CONSTANTS, DEFAULT_CONSTANTS_24, FAN_REGISTRY,
)
from ml_model import (
    train_all_models, predict_performance,
    find_best_operating_point, find_motor_recommendation,
    TARGET_COLS, STANDARD_MOTORS,
)
from plots import (
    create_fan_curve, create_ftp_curve, create_power_curve,
    create_efficiency_curves, create_combined_performance,
    create_angle_comparison, create_3d_surface,
    create_prediction_vs_actual, create_ml_prediction_curves,
    create_system_resistance_overlay, get_angle_color,
)

# ── Extension layer ────────────────────────────────────────────
from app_extensions import render_sidebar_mode_selector, render_extension_page
from fan_db import init_db, list_fans as db_list_fans, get_raw_df, get_fan_constants
from model_store import get_or_train_model, is_model_stale

init_db()   # creates tables + seeds TA18 / TA24 on first run

# Build a fan_id lookup from display_name (used for model_store calls)
def _fan_id_from_name(display_name: str) -> str:
    return (
        display_name.lower()
        .replace('"', "in").replace("'", "")
        .replace(" ", "_").strip("_")
    )

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title='Fan Performance Tool — 18" & 24"',
    page_icon='🌀',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    color: #fff; font-size: 2rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #00D4FF, #00FF85);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p {
    color: rgba(255,255,255,0.65); font-size: 1rem; margin: .5rem 0 0 0;
}

.metric-card {
    background: linear-gradient(135deg, rgba(15,12,41,.8), rgba(48,43,99,.55));
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 12px; padding: 1.2rem; text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,.2);
}
.metric-card h3 {
    color: rgba(255,255,255,.55); font-size: .78rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 1px; margin: 0;
}
.metric-card .value {
    color: #00D4FF; font-size: 1.7rem; font-weight: 700; margin: .3rem 0;
}
.metric-card .unit {
    color: rgba(255,255,255,.4); font-size: .72rem;
}

.info-badge {
    background: rgba(0,212,255,.12); border: 1px solid rgba(0,212,255,.28);
    border-radius: 8px; padding: .75rem 1.1rem; color: #00D4FF;
    font-size: .88rem; margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1a3e);
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"]      { border-radius: 8px 8px 0 0; padding: 10px 20px; }
.stDataFrame                       { border-radius: 8px; overflow: hidden; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# SIDEBAR — configuration
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    mode = render_sidebar_mode_selector()

if mode != "⚙️  Fan Analysis":
    render_extension_page(mode)
    st.stop()

with st.sidebar:
    # ── existing Configuration section (unchanged below) ──────
    st.markdown('### ⚙️ Configuration')
    st.markdown('---')

    # ── Fan selector ───────────────────────────────────────────
    st.markdown('### 🌀 Fan Selection')
    fan_options = list(FAN_REGISTRY.keys())
    selected_fan = st.selectbox('Select Fan', fan_options, key='fan_select')
    _, _def_ct = FAN_REGISTRY[selected_fan]

    st.markdown('### 📐 Test Parameters')
    duct_dia = st.number_input('Duct Diameter (m)',    value=_def_ct['duct_dia_m'],      format='%.4f', step=0.001)
    cd       = st.number_input('Discharge Coeff (CD)', value=_def_ct['discharge_coeff'], format='%.2f', step=0.01)
    cw       = st.number_input('Wattmeter Corr (CW)',  value=_def_ct['cw'],              format='%.1f', step=1.0)

    st.markdown('### 🌡️ Conditions')
    test_temp   = st.number_input('Test Temp (°C)',   value=int(_def_ct['test_temp_c']),      step=1)
    test_baro   = st.number_input('Test Baro (mm Hg)', value=int(_def_ct['test_baro_mmhg']),   step=1)
    design_temp = st.number_input('Design Temp (°C)', value=int(_def_ct['design_temp_c']),    step=1)
    design_baro = st.number_input('Design Baro (mm Hg)', value=int(_def_ct['design_baro_mmhg']), step=1)

    st.markdown('### ⚡ Motor')
    design_speed = st.number_input('Design Speed (RPM)', value=int(_def_ct['design_speed_rpm']), step=1)
    motor_eff    = st.slider('Motor Efficiency (%)', 50, 95,
                             int(_def_ct['motor_efficiency'] * 100)) / 100.0

    constants = dict(
        duct_dia_m=duct_dia, discharge_coeff=cd, cw=cw,
        test_temp_c=test_temp, test_baro_mmhg=test_baro,
        design_temp_c=design_temp, design_baro_mmhg=design_baro,
        design_speed_rpm=design_speed, motor_efficiency=motor_eff,
        g=9.81,
    )


# ────────────────────────────────────────────────────────────────
# SESSION STATE — editable raw data (persists across reruns)
# ────────────────────────────────────────────────────────────────
_skey = f'edited_{selected_fan}'
if _skey not in st.session_state:
    st.session_state[_skey] = get_raw_data(selected_fan)

raw_df = st.session_state[_skey]

# ────────────────────────────────────────────────────────────────
# DATA + MODELS  (cached — key includes df content + constants)
# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Computing derived quantities …')
def _compute(fan, ct, df_json):
    import io
    raw = pd.read_json(io.StringIO(df_json))
    return compute_derived_quantities(df=raw, fan=fan, constants=dict(ct))

@st.cache_resource(show_spinner='Training ML models (LOOCV) …')
def _train(fan, ct, df_json):
    import io
    raw = pd.read_json(io.StringIO(df_json))
    df  = compute_derived_quantities(df=raw, fan=fan, constants=dict(ct))
    return train_all_models(df)

# _train replaced by model_store — only retrains when data changed
ct = tuple(sorted(constants.items()))
df_json = raw_df.to_json()
df = _compute(selected_fan, ct, df_json)

try:
    _fan_id = _fan_id_from_name(selected_fan)
    mi = get_or_train_model(
        fan_id=_fan_id,
        df_computed=df,
        force_retrain=False,
    )
except Exception as _train_err:
    st.error(f'❌ ML training failed: {_train_err}')
    if st.button('🔄 Clear Cache & Retry'):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
fan_inches = '18"' if '18' in selected_fan else '24"'
st.markdown(f"""
<div class="main-header">
  <h1>🌀 {fan_inches} Tube Axial Fan — Performance Analysis</h1>
  <p>ML-Powered Performance Prediction &amp; Engineering Visualisation Tool</p>
</div>""", unsafe_allow_html=True)

# key metrics banner
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('📏 Fan Size',     fan_inches,              f'{constants["duct_dia_m"]*1000:.0f} mm')
c2.metric('🔄 Design RPM',   f'{constants["design_speed_rpm"]}', 'RPM')
c3.metric('💨 Max Volume',   f'{df["Q_CMH"].max():.0f}',         'CMH')
c4.metric('📊 Max FSP',      f'{df["FSP"].max():.1f}',           'mm WG')
c5.metric('🎯 Peak η',       f'{df["Total_Eff"].max():.1f}%',    'Total')


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    '📋 Data & Calculations',
    '📈 Performance Curves',
    '🤖 ML Predictions',
    '🎯 Fan Selection',
])

# ── TAB 1 ──────────────────────────────────────────────────────
with tab1:
    st.markdown('### 📋 Raw Test Data — Live Editor')
    st.markdown(
        '<div class="info-badge">✏️ Edit cells directly, add or delete rows, '
        'then click <strong>Apply Changes</strong> to recompute everything.</div>',
        unsafe_allow_html=True)

    edited = st.data_editor(
        raw_df,
        num_rows='dynamic',          # enables + Add row / trash-can delete
        use_container_width=True,
        height=360,
        key=f'editor_{selected_fan}',
    )

    col_apply, col_reset, _ = st.columns([1, 1, 4])
    if col_apply.button('✅ Apply Changes & Retrain', type='primary', use_container_width=True):
        st.session_state[_skey] = edited.reset_index(drop=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if col_reset.button('🔄 Reset to Original Data', use_container_width=True):
        st.session_state[_skey] = get_raw_data(selected_fan)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown('---')
    st.markdown('### 🔧 Computed Quantities')

    show_cols = [
        'Srno', 'ANGLE', 'DEL_P', 'SP', 'WT', 'Mi_kW', 'PF',
        'Qt_CMH', 'V_out_mps', 'VPot', 'VPi', 'TPot', 'TPi',
        'FTPT', 'Mo_kW', 'Q_CMH', 'R_VPo', 'FTP', 'FSP', 'BKW',
        'Air_Power_ST', 'Air_Power_T', 'Static_Eff', 'Total_Eff',
        'Motor_Input',
    ]
    show_df = df[show_cols].round(4)
    st.dataframe(show_df, use_container_width=True, height=400)
    st.download_button('📥 Download CSV', show_df.to_csv(index=False),
                       'fan_computed_data.csv', 'text/csv')

    st.markdown('---')
    with st.expander('📖 Calculation Reference'):
        st.markdown("""
| # | Quantity | Formula |
|---|---------|---------|
| 1 | WT | `1.205 × (B + 0.0737×SP) / 760 × 293 / (273+Ts)` |
| 2 | WTd | `1.205 × B_d / 760 × 293 / (273+T_d)` |
| 3 | Mi (kW) | `(W1+W2) × CW / 1000` |
| 4 | PF | `Mi×1000 / (√3 × V × I)` |
| 5 | Qt (CMH) | `12500 × CD × D² × √(DP/WT)` |
| 6 | V_out (m/s) | `Qt / (A × 3600)` |
| 7 | VPot | `(V²/2g) × WT` |
| 8 | VPi | `(V_in_mhr / 16000)² × WT` |
| 9 | TPot | `SP + VPot` |
| 10 | TPi | `VPi` (free inlet) |
| 11 | FTPT | `TPot − TPi` |
| 12 | Mo | `Mi × Motor_Eff` |
| 13 | Q (rated CMH) | `Qt × (N/Nt)` |
| 14 | FTP | `FTPT × (N/Nt)² × (WTd/WT)` |
| 15 | FSP | `FTP − R.VPo` |
| 16 | BKW | `Mo × (N/Nt)³ × (WTd/WT)` |
| 17 | η_static | `2.723 × Q × FSP × 10⁻⁶ / BKW × 100` |
| 18 | η_total | `2.723 × Q × FTP × 10⁻⁶ / BKW × 100` |
""")


# ── TAB 2 ──────────────────────────────────────────────────────
with tab2:
    st.markdown('### 📈 Standard Performance Curves')
    st.markdown('<div class="info-badge">💡 All charts are interactive — '
                'hover, zoom, pan, and compare angles.</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(create_fan_curve(df),  use_container_width=True)
    col_b.plotly_chart(create_ftp_curve(df),  use_container_width=True)

    st.plotly_chart(create_power_curve(df),        use_container_width=True)
    st.plotly_chart(create_efficiency_curves(df),   use_container_width=True)

    st.markdown('---')
    st.markdown('### 📊 Angle-wise Comparison')
    fig_cmp, sdf = create_angle_comparison(df)
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.dataframe(sdf.set_index('Angle'), use_container_width=True)

    st.markdown('---')
    st.markdown('### 📈 Combined Performance (Single Angle)')
    angles_sorted = sorted(df['ANGLE'].unique())
    sel_angle = st.selectbox('Select Blade Angle', angles_sorted,
                              index=min(2, len(angles_sorted) - 1),
                              key=f'comb_angle_{selected_fan}')
    st.plotly_chart(create_combined_performance(df, sel_angle),
                    use_container_width=True)

    st.markdown('---')
    st.markdown('### 🔄 System Resistance Overlay')
    sr1, sr2 = st.columns([1, 3])
    with sr1:
        sr_angle = st.selectbox('Blade Angle', sorted(df['ANGLE'].unique()),
                                key=f'sr_angle_{selected_fan}')
        k_sys = st.number_input('Resistance k', value=1e-6, format='%.2e',
                                help='SP = k × Q²')
    with sr2:
        st.plotly_chart(create_system_resistance_overlay(df, sr_angle, k_sys),
                        use_container_width=True)

    st.markdown('---')
    st.markdown('### 🌐 3D Surface Plots')
    surf_t = st.selectbox('Parameter', ['FSP', 'FTP', 'BKW', 'Static_Eff', 'Total_Eff'])
    st.plotly_chart(create_3d_surface(df, surf_t), use_container_width=True)


# ── TAB 3 ──────────────────────────────────────────────────────
with tab3:
    st.markdown('### 🤖 Machine Learning Model Performance')

    best = mi['best_model_name']
    mcols = st.columns(3)
    for i, (name, res) in enumerate(mi['results'].items()):
        with mcols[i]:
            badge  = '🏆 BEST' if name == best else ''
            border = 'rgba(0,255,133,.4)' if name == best else 'rgba(255,255,255,.1)'
            st.markdown(f"""
<div class="metric-card" style="border-color:{border}">
  <h3>{name} {badge}</h3>
  <div class="value">{res['avg_r2_cv']:.3f}</div>
  <div class="unit">Avg LOOCV R²</div>
</div>""", unsafe_allow_html=True)

    st.markdown('')

    with st.expander('📊 Detailed Model Metrics (LOOCV)'):
        for name, res in mi['results'].items():
            st.markdown(f"**{name}** {'🏆' if name == best else ''}")
            mdf = pd.DataFrame({
                'Target':     TARGET_COLS,
                'Train R²':   [res['r2_train'][t] for t in TARGET_COLS],
                'CV R²':      [res['r2_cv'][t]    for t in TARGET_COLS],
                'Train RMSE': [res['rmse_train'][t] for t in TARGET_COLS],
                'CV RMSE':    [res['rmse_cv'][t]    for t in TARGET_COLS],
            })
            st.dataframe(mdf.round(4), use_container_width=True, hide_index=True)
            st.markdown('')

    st.markdown('---')
    st.markdown('### 🎯 Predicted vs Actual (LOOCV)')
    pva_t = st.selectbox('Target variable', TARGET_COLS,
                          index=TARGET_COLS.index('Q_CMH'), key='pva')
    st.plotly_chart(create_prediction_vs_actual(df, mi, pva_t),
                    use_container_width=True)

    st.markdown('---')
    st.markdown('### 🔮 Predict at Custom Blade Angle')

    pc1, pc2 = st.columns([1, 3])
    with pc1:
        c_angle = st.slider('Blade Angle (°)', 15.0, 50.0, 32.5, 0.5,
                             key=f'custom_angle_{selected_fan}')
        st.markdown(f"""
<div class="metric-card">
  <h3>Predicting for</h3>
  <div class="value" style="color:#FF6BFF">{c_angle}°</div>
  <div class="unit">Blade Angle</div>
</div>""", unsafe_allow_html=True)
        show_act = st.checkbox('Show actual data', value=True)

    with pc2:
        prd = predict_performance(mi, c_angle)
        st.plotly_chart(
            create_ml_prediction_curves(prd, df if show_act else None, c_angle),
            use_container_width=True)

    with st.expander('📋 View Predicted Data'):
        p_show = prd[['ANGLE', 'DEL_P', 'SP', 'Q_CMH', 'FSP',
                       'FTP', 'BKW', 'Static_Eff', 'Total_Eff']].round(2)
        st.dataframe(p_show, use_container_width=True, hide_index=True)
        st.download_button('📥 Download Predictions', p_show.to_csv(index=False),
                           f'predicted_{c_angle}deg.csv', 'text/csv')


# ── TAB 4 ──────────────────────────────────────────────────────
with tab4:
    st.markdown('### 🎯 Fan Selection — Find Optimal Operating Point')
    st.markdown('<div class="info-badge">Enter your system requirements. '
                'The tool recommends the best <strong>blade angle</strong> '
                'and <strong>motor speed</strong> across all standard catalogue '
                'options (950 / 1440 / 2850 RPM).</div>',
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    req_cmh = s1.number_input('Required Volume (CMH)', 100, 50000, 5000, 100)
    req_sp  = s2.number_input('Required SP (mm WG)',    0.0, 60.0,  10.0, 0.5)
    find    = s3.button('🔍 Find Best Configuration', type='primary',
                        use_container_width=True)

    if find:
        design_rpm = constants['design_speed_rpm']

        with st.spinner('🔄 Evaluating all motor options via fan laws …'):
            ba, bp, md = find_best_operating_point(mi, req_cmh, req_sp)
            motor_recs = find_motor_recommendation(mi, req_cmh, req_sp, design_rpm)

        st.markdown('---')
        if md < 0.3:
            st.success(f'✅ At design speed ({design_rpm} RPM): good match, deviation {md:.1%}')
        else:
            st.warning(f'⚠️ At design speed ({design_rpm} RPM): deviation {md:.1%}. A different motor may fit better.')

        # ── Motor cards ──────────────────────────────────────────
        st.markdown('### 🔌 Motor Selection — All Standard Options')
        st.markdown(
            '<div class="info-badge">Fan laws: Q∝N, SP∝N², P∝N³. '
            'Efficiency is speed-independent. Cards ranked best-to-worst match.</div>',
            unsafe_allow_html=True)

        mcols = st.columns(3)
        for i, rec in enumerate(motor_recs):
            m, sc, dev = rec['motor'], rec['scaled'], rec['deviation']
            is_best   = rec['recommended']
            border    = 'rgba(0,255,133,.45)' if is_best else 'rgba(255,255,255,.12)'
            val_color = '#00FF85' if is_best else '#00D4FF'
            badge     = '🏆 BEST MATCH' if is_best else f'#{i+1}'
            match_lbl = ('Excellent' if dev < 0.2 else 'Good' if dev < 0.4 else 'Fair' if dev < 0.7 else 'Poor')
            dev_color = '#00FF85' if dev < 0.2 else ('#FFD700' if dev < 0.5 else '#FF4444')

            with mcols[i]:
                _angle_str = f"{rec['angle']}°"
                st.markdown(f"""
<div class="metric-card" style="border-color:{border};padding:1.4rem;text-align:left">
  <div style="text-align:center;margin-bottom:.6rem">
    <span style="font-size:.75rem;font-weight:600;color:rgba(255,255,255,.5);text-transform:uppercase;letter-spacing:1px">{badge}</span>
  </div>
  <div style="text-align:center;margin-bottom:.8rem">
    <div style="font-size:1.8rem;font-weight:700;color:{val_color}">{m['rpm']} RPM</div>
    <div style="font-size:.82rem;color:rgba(255,255,255,.45)">{m['poles']}-Pole Induction Motor</div>
  </div>
  <hr style="border-color:rgba(255,255,255,.1);margin:.5rem 0">
  <table style="width:100%;font-size:.83rem;color:#E0E0E0">
    <tr><td>Blade Angle</td>    <td style="text-align:right;color:#FF6BFF"><b>{_angle_str}</b></td></tr>
    <tr><td>Volume Flow</td>    <td style="text-align:right"><b>{sc['Q_CMH']:.0f} CMH</b></td></tr>
    <tr><td>Static Press.</td>  <td style="text-align:right"><b>{sc['FSP']:.1f} mm WG</b></td></tr>
    <tr><td>Total Press.</td>   <td style="text-align:right"><b>{sc['FTP']:.1f} mm WG</b></td></tr>
    <tr><td>Shaft Power</td>    <td style="text-align:right"><b>{sc['BKW']*1000:.0f} W</b></td></tr>
    <tr><td>&eta; Static</td>   <td style="text-align:right"><b>{sc['Static_Eff']:.1f}%</b></td></tr>
    <tr><td>&eta; Total</td>    <td style="text-align:right"><b>{sc['Total_Eff']:.1f}%</b></td></tr>
  </table>
  <hr style="border-color:rgba(255,255,255,.1);margin:.5rem 0">
  <div style="text-align:center;color:{dev_color};font-size:.82rem;font-weight:600">
    {match_lbl} &mdash; &Delta; {dev:.1%} from target
  </div>
</div>""", unsafe_allow_html=True)

        # ── Per-motor expandable detail ───────────────────────────
        st.markdown('')
        st.markdown('#### 📈 Performance Curves per Motor Option')
        for rec in motor_recs:
            m, sc = rec['motor'], rec['scaled']
            _pfx  = '🏆 ' if rec['recommended'] else ''
            label = f"{_pfx}{m['label']} — Blade {rec['angle']}°"
            with st.expander(label, expanded=rec['recommended']):
                ic = st.columns(4)
                ic[0].metric('Volume',          f"{sc['Q_CMH']:.0f} CMH",
                             f"{sc['Q_CMH']-req_cmh:+.0f} vs target")
                ic[1].metric('Static Pressure', f"{sc['FSP']:.1f} mm WG",
                             f"{sc['FSP']-req_sp:+.2f} vs target")
                ic[2].metric('Shaft Power',     f"{sc['BKW']*1000:.0f} W")
                ic[3].metric('η Total', f"{sc['Total_Eff']:.1f}%")
                st.plotly_chart(
                    create_ml_prediction_curves(predict_performance(mi, rec['angle']), df, rec['angle']),
                    use_container_width=True,
                    key=f"motor_curve_{m['rpm']}_{rec['angle']}")

        # ── Comparison table ─────────────────────────────────────
        st.markdown('---')
        st.markdown('#### 📋 Side-by-Side Comparison')
        tbl = []
        for rec in motor_recs:
            m, sc = rec['motor'], rec['scaled']
            _match_icon = '✅' if rec['deviation'] < 0.3 else '⚠️'
            tbl.append({
                'Motor':           m['label'],
                'Blade Angle (°)': rec['angle'],
                'Volume (CMH)':    round(sc['Q_CMH']),
                'vs Required CMH': f"{sc['Q_CMH']-req_cmh:+.0f}",
                'FSP (mm WG)':     round(sc['FSP'], 2),
                'vs Required SP':  f"{sc['FSP']-req_sp:+.2f}",
                'Power (W)':       round(sc['BKW'] * 1000, 1),
                'η Static (%)':    round(sc['Static_Eff'], 1),
                'η Total (%)':     round(sc['Total_Eff'], 1),
                'Match':           f"{_match_icon} {rec['deviation']:.1%}",
                'Best':            '🏆' if rec['recommended'] else '',
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    # BEP table (always visible)
    st.markdown('---')
    st.markdown('### 🏆 Best Efficiency Points — All Angles')
    bep_rows = []
    for angle in sorted(df['ANGLE'].unique()):
        ad = df[df['ANGLE'] == angle]
        bi = ad['Total_Eff'].idxmax()
        br = ad.loc[bi]
        bep_rows.append({
            'Angle (°)':      angle,
            'Volume at BEP':  round(br['Q_CMH']),
            'FSP at BEP':     round(br['FSP'], 2),
            'FTP at BEP':     round(br['FTP'], 2),
            'Static Eff (%)': round(br['Static_Eff'], 1),
            'Total Eff (%)':  round(br['Total_Eff'], 1),
            'BKW (W)':        round(br['BKW'] * 1000, 1),
        })
    st.dataframe(pd.DataFrame(bep_rows), use_container_width=True, hide_index=True)


# ── FOOTER ─────────────────────────────────────────────────────
st.markdown('---')
st.markdown(
    '<div style="text-align:center;color:rgba(255,255,255,.25);font-size:.78rem">'
    '🌀 Tube Axial Fan Performance Tool (18" & 24") — ML-Powered Engineering Analysis'
    '</div>', unsafe_allow_html=True)
