"""
app.py — Tube Axial Fan Performance Analysis Tool (18" & 24")

A Streamlit dashboard with four tabs:
  1. Raw Data & Calculations
  2. Performance Curves
  3. ML Predictions
  4. Fan Selection
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from data import (
    get_raw_data, compute_derived_quantities,
    DEFAULT_CONSTANTS, DEFAULT_CONSTANTS_24, FAN_REGISTRY,
)
from physics_model import (
    predict_performance,
    find_best_operating_point, find_motor_recommendation,
    select_motor_rating,
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
from app_extensions import (
    render_sidebar_mode_selector,
    render_extension_page,
    render_company_header_html,
    get_logo_base64,
)
from fan_db import init_db, list_fans as db_list_fans, get_raw_df, get_fan_constants

# ── Unit Conversion Helpers ────────────────────────────────────
CMH_TO_CFM = 0.588577779
CFM_TO_CMH = 1.69901082

def convert_flow_out(cmh):
    if st.session_state.get('flow_unit', 'CMH') == 'CFM':
        return cmh * CMH_TO_CFM
    return cmh

def convert_flow_in(user_input):
    if st.session_state.get('flow_unit', 'CMH') == 'CFM':
        return user_input * CFM_TO_CMH
    return user_input

def flow_unit_label():
    return st.session_state.get('flow_unit', 'CMH')

init_db()   # creates tables + seeds TA18 / TA24 on first run

# Build a fan_id lookup from display_name (used for model_store calls)
def _fan_id_from_name(display_name: str) -> str:
    return (
        display_name.lower()
        .replace('"', "in").replace("'", "")
        .replace(" ", "_").strip("_")
    )

# ── Page config ────────────────────────────────────────────────
_logo_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'logo.png')
st.set_page_config(
    page_title='Maxim Air — Tube Axial Fan Performance Tool',
    page_icon=_logo_file if os.path.exists(_logo_file) else '🌀',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #FFFFFF !important;
    color: #0F2A28 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #0F2A28 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* Paragraphs & general markdown */
p, span, div, li, .stMarkdown p, .stMarkdown span {
    color: #0F2A28;
}

/* Form Labels & Widget Labels */
label, label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] span {
    color: #0F2A28 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Inputs, Selectboxes, Number Inputs */
input, textarea, div[data-baseweb="select"], div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    color: #0F2A28 !important;
    border-color: #B2DFDB !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus, div[data-baseweb="select"]:focus-within, div[data-baseweb="input"]:focus-within {
    border-color: #00897B !important;
    box-shadow: 0 0 0 2px rgba(0, 137, 123, 0.2) !important;
}

/* Buttons */
button[kind="primary"], .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #00897B 0%, #007367 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
    box-shadow: 0 2px 8px rgba(0, 137, 123, 0.25) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #007367 0%, #005B52 100%) !important;
    box-shadow: 0 4px 12px rgba(0, 137, 123, 0.35) !important;
}
button[kind="secondary"], .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background-color: #F8FAF9 !important;
    color: #0F2A28 !important;
    border: 1.5px solid #B2DFDB !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {
    background-color: #E0F2F1 !important;
    border-color: #00897B !important;
    color: #00897B !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F8FAF9 !important;
    border-right: 1px solid #E0EFEF !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #0F2A28 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E0EFEF;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    color: #3B5957 !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: #00897B !important;
    border-bottom: 2.5px solid #00897B !important;
    font-weight: 700 !important;
}

/* Metrics */
div[data-testid="stMetricValue"] > div {
    color: #00897B !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] > div > p {
    color: #3B5957 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px !important;
}

/* Tables & Dataframes */
.stDataFrame, div[data-testid="stTable"] {
    border: 1px solid #D1E7E5 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #D1E7E5 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(15, 42, 40, 0.03) !important;
}
div[data-testid="stExpander"] summary span {
    color: #0F2A28 !important;
    font-weight: 600 !important;
}

/* Main Headers & Company Head */
.companyhead-banner {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    background: linear-gradient(135deg, #09201E 0%, #00564D 50%, #00897B 100%);
    padding: 1.3rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(0, 137, 123, 0.3);
    box-shadow: 0 6px 24px rgba(0, 137, 123, 0.18);
}
.companyhead-logo-container {
    flex-shrink: 0;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
}
.companyhead-logo-img {
    width: 60px;
    height: 60px;
    object-fit: contain;
    display: block;
    filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.3));
}
.companyhead-logo-fallback {
    font-size: 2.2rem;
    line-height: 1;
}
.companyhead-content {
    flex-grow: 1;
    min-width: 0;
}
.companyhead-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    color: #E0F2F1;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 4px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.companyhead-title {
    color: #FFFFFF !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
}
.companyhead-subtitle {
    color: rgba(255, 255, 255, 0.88) !important;
    font-size: 0.88rem !important;
    margin: 0.25rem 0 0 0 !important;
    line-height: 1.4 !important;
}

.main-header {
    background: linear-gradient(135deg, #0F2A28 0%, #00897B 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 137, 123, 0.2);
    box-shadow: 0 6px 24px rgba(0, 137, 123, 0.15);
}
.main-header h1 {
    color: #FFFFFF !important;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 0.95rem;
    margin: 0.4rem 0 0 0;
    line-height: 1.4;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #D1E7E5;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(15, 42, 40, 0.04);
}
.metric-card h3 {
    color: #3B5957 !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0;
}
.metric-card .value {
    color: #00897B;
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.metric-card .unit {
    color: #537775;
    font-size: 0.75rem;
}

.info-badge {
    background: #E0F2F1;
    border: 1px solid #80CBC4;
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    color: #0F2A28;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

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
    db_fans = db_list_fans()
    if not db_fans:
        st.error("No fans found in database.")
        st.stop()
        
    fan_display_to_id = {f["display_name"]: f["fan_id"] for f in db_fans}
    fan_options = list(fan_display_to_id.keys())
    
    selected_display_name = st.selectbox('Select Fan', fan_options, key='fan_select')
    selected_fan = fan_display_to_id[selected_display_name]
    _db_constants = get_fan_constants(selected_fan)

    st.markdown('### 📐 Test Parameters')
    duct_dia = st.number_input('Duct Diameter (m)',    value=_db_constants['duct_dia_m'],      format='%.4f', step=0.001)
    cd       = st.number_input('Discharge Coeff (CD)', value=_db_constants['discharge_coeff'], format='%.2f', step=0.01)
    cw       = st.number_input('Wattmeter Corr (CW)',  value=_db_constants['cw'],              format='%.1f', step=1.0)

    st.markdown('### 🌡️ Conditions')
    test_temp   = st.number_input('Test Temp (°C)',   value=int(_db_constants['test_temp_c']),      step=1)
    test_baro   = st.number_input('Test Baro (mm Hg)', value=int(_db_constants['test_baro_mmhg']),   step=1)
    design_temp = st.number_input('Design Temp (°C)', value=int(_db_constants['design_temp_c']),    step=1)
    design_baro = st.number_input('Design Baro (mm Hg)', value=int(_db_constants['design_baro_mmhg']), step=1)

    st.markdown('### ⚡ Motor')
    design_speed = st.number_input('Design Speed (RPM)', value=int(_db_constants['design_speed_rpm']), step=1)
    motor_eff    = st.slider('Motor Efficiency (%)', 50, 95,
                             int(_db_constants['motor_efficiency'] * 100)) / 100.0

    constants = dict(
        duct_dia_m=duct_dia, discharge_coeff=cd, cw=cw,
        test_temp_c=test_temp, test_baro_mmhg=test_baro,
        design_temp_c=design_temp, design_baro_mmhg=design_baro,
        design_speed_rpm=design_speed, motor_efficiency=motor_eff,
        poles=_db_constants.get('poles', 4),
        g=9.81,
    )

    st.markdown('')
    if st.button('💾 Save Parameters to DB', type='primary', use_container_width=True, help='Save CW (Wattmeter Corr) and test parameters permanently for this fan'):
        from fan_db import save_constants
        save_constants(selected_fan, constants)
        st.success('✅ Parameters permanently saved to DB!')
        st.rerun()



# ────────────────────────────────────────────────────────────────
# SESSION STATE — editable raw data (persists across reruns)
# ────────────────────────────────────────────────────────────────
_skey = f'edited_{selected_fan}'
if _skey not in st.session_state:
    st.session_state[_skey] = get_raw_df(selected_fan)

raw_df = st.session_state[_skey]

# ────────────────────────────────────────────────────────────────
# DATA + MODELS  (cached — key includes df content + constants)
# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Computing derived quantities …')
def _compute(fan, ct, df_json):
    import io
    raw = pd.read_json(io.StringIO(df_json))
    return compute_derived_quantities(df=raw, fan=fan, constants=dict(ct))

# Compute logic (determinisic physics, no ML training required)
ct = tuple(sorted(constants.items()))
df_json = raw_df.to_json()
df = _compute(selected_fan, ct, df_json)

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown(
    render_company_header_html(
        title=f"{selected_display_name} — Performance Analysis",
        subtitle="Physics-Based Performance Modeling, Interpolation & Visualisation Tool",
        badge="MAXIM AIR • FAN PERFORMANCE ANALYSIS",
    ),
    unsafe_allow_html=True,
)

# key metrics banner
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('📏 Fan Name',     selected_display_name,              f'{constants["duct_dia_m"]*1000:.0f} mm')
c2.metric('🔄 Design RPM',   f'{constants["design_speed_rpm"]}', 'RPM')
c3.metric('💨 Max Volume',   f'{convert_flow_out(df["Q_CMH"].max()):.0f}', flow_unit_label())
c4.metric('📊 Max FSP',      f'{df["FSP"].max():.1f}',           'mm WG')
c5.metric('🎯 Peak η',       f'{df["Total_Eff"].max():.1f}%',    'Total')


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    '📋 Data & Calculations',
    '📈 Performance Curves',
    '🔮 Custom Interpolation',
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
        st.session_state[_skey] = get_raw_df(selected_fan)
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
    show_df = df[show_cols].copy()
    unit = flow_unit_label()
    if unit == 'CFM':
        show_df['Qt_CMH'] = show_df['Qt_CMH'] * CMH_TO_CFM
        show_df['Q_CMH'] = show_df['Q_CMH'] * CMH_TO_CFM
        show_df = show_df.rename(columns={'Qt_CMH': 'Qt_CFM', 'Q_CMH': 'Q_CFM'})
    show_df = show_df.round(4)
    st.dataframe(show_df, use_container_width=True, height=400)
    st.download_button('📥 Download CSV', show_df.to_csv(index=False),
                       'fan_computed_data.csv', 'text/csv')

    st.markdown('---')
    with st.expander('📖 Calculation Reference'):
        st.markdown("""
| # | Quantity | Formula |
|---|---------|---------|| 1 | WT | `1.205 × (B + 0.0737×SP) / 760 × 293 / (273+Ts)` |
| 2 | WTd | `1.205 × B_d / 760 × 293 / (273+T_d)` |
| 3 | Mi (kW) | `(W1+W2) × CW / 1000` |
| 4 | PF | `Mi×1000 / (√3 × V × I)` |
| 5 | Qt (CMH) | `12500 × CD × D² × √(DP/WT)` |
| 6 | V_out (m/s) | `Qt / (A × 3600)` |
| 7 | VPot | `(V_out_mps² / 2g) × WT` |
| 8 | VPi | `(V_in_mps² / 2g) × WT`  *(exact — same as VPot; inlet = outlet area)* |
| 9 | TPot | `SP + VPot` |
| 10 | TPi | `VPi` (free inlet, SPi = 0) |
| 11 | FTPT | `TPot − TPi = SP`  *(VPot = VPi for same-area fan)* |
| 12 | Mo | `Mi × Motor_Eff` |
| 13 | Q (rated CMH) | `Qt × (N/Nt)` |
| 14 | R_VPo | `(V_rated_mps² / 2g) × WTd`  *(exact rated outlet VP)* |
| 15 | FTP | `FTPT × (N/Nt)² × (WTd/WT)` |
| 16 | FSP | `FTP − R_VPo`  *(may be ≤ 0 at free-delivery points — physically correct)* |
| 17 | BKW | `Mo × (N/Nt)³ × (WTd/WT)` |
| 18 | η_static | `2.725 × Q × max(FSP, 0) × 10⁻⁶ / BKW × 100`  *(FSP floored at 0)* |
| 19 | η_total | `2.725 × Q × FTP × 10⁻⁶ / BKW × 100` |
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
    st.markdown('### 🔮 Interpolate at Custom Blade Angle')
    st.markdown('<div class="info-badge">Uses deterministic polynomial regression mapped between the nearest tested angles.</div>', unsafe_allow_html=True)
    st.markdown('---')

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
        prd = predict_performance(df, c_angle)
        st.plotly_chart(
            create_ml_prediction_curves(prd, df if show_act else None, c_angle),
            use_container_width=True)

    with st.expander('📋 View Predicted Data'):
        p_show = prd[['ANGLE', 'Q_CMH', 'SP', 'FSP',
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
    unit = flow_unit_label()
    if unit == 'CFM':
        req_val = s1.number_input('Required Volume (CFM)', int(100 * CMH_TO_CFM), int(50000 * CMH_TO_CFM), int(5000 * CMH_TO_CFM), int(100 * CMH_TO_CFM))
        req_cmh = req_val * CFM_TO_CMH
    else:
        req_cmh = s1.number_input('Required Volume (CMH)', 100, 50000, 5000, 100)
    req_sp  = s2.number_input('Required SP (mm WG)',    0.0, 60.0,  10.0, 0.5)
    find    = s3.button('🔍 Find Best Configuration', type='primary',
                        use_container_width=True)

    if find:
        design_rpm = constants['design_speed_rpm']

        with st.spinner('🔄 Evaluating all motor options via fan laws …'):
            ba, bp, md = find_best_operating_point(df, req_cmh, req_sp)
            motor_recs = find_motor_recommendation(df, req_cmh, req_sp, design_rpm)

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

        # ── Exact quantities from required operating point (no ML) ────
        _sel_area    = np.pi / 4 * constants["duct_dia_m"]**2
        _v_out_req   = req_cmh / (_sel_area * 3600)          # m/s
        _v_out_req_mhr = _v_out_req * 3600                   # m/hr
        # Total pressure at required point (user formula):
        #   FTP_req = SP_req + (V_out_req_mhr / 16000)² × 1.2
        _ftp_req = req_sp + (_v_out_req_mhr / 16000)**2 * 1.2

        for i, rec in enumerate(motor_recs):
            m, sc, dev = rec['motor'], rec['scaled'], rec['deviation']
            m_rat     = rec['motor_rating']
            is_best   = rec['recommended']
            border    = '#00897B' if is_best else 'rgba(0, 137, 123, 0.22)'
            val_color = '#00897B' if is_best else '#0F2A28'
            badge_bg  = '#E0F2F1' if is_best else '#F4F8F7'
            badge_clr = '#00897B' if is_best else '#4A6966'
            badge     = '🏆 BEST MATCH' if is_best else f'#{i+1}'
            match_lbl = ('Excellent' if dev < 0.2 else 'Good' if dev < 0.4 else 'Fair' if dev < 0.7 else 'Poor')
            dev_color = '#059669' if dev < 0.2 else ('#D97706' if dev < 0.5 else '#DC2626')

            with mcols[i]:
                _angle_str = f"{rec['angle']}°"
                # V_out and FTP are pure geometry + given constants — exact, no ML
                _v_out_card = _v_out_req   # same area → same velocity for all cards
                st.markdown(f"""
<div class="metric-card" style="border: 1.5px solid {border};padding:1.4rem;text-align:left;background:#FFFFFF">
  <div style="text-align:center;margin-bottom:.6rem">
    <span style="font-size:.75rem;font-weight:700;color:{badge_clr};background:{badge_bg};padding:0.2rem 0.6rem;border-radius:12px;text-transform:uppercase;letter-spacing:1px">{badge}</span>
  </div>
  <div style="text-align:center;margin-bottom:.8rem">
    <div style="font-size:1.8rem;font-weight:800;color:{val_color}">{m['rpm']} RPM</div>
    <div style="font-size:.82rem;color:#4A6966;font-weight:500">{m['poles']}-Pole Induction Motor</div>
  </div>
  <hr style="border:none;border-top:1px solid rgba(0,137,123,0.15);margin:.5rem 0">
  <table style="width:100%;font-size:.83rem;color:#0F2A28">
    <tr><td>Blade Angle</td>      <td style="text-align:right;color:#00897B"><b>{_angle_str}</b></td></tr>
    <tr><td>Volume Flow</td>      <td style="text-align:right"><b>{convert_flow_out(sc['Q_CMH']):.0f} {unit}</b></td></tr>
    <tr><td>Outlet Velocity</td>  <td style="text-align:right"><b>{_v_out_card:.2f} m/s</b></td></tr>
    <tr><td>Static Press.</td>    <td style="text-align:right"><b>{sc['FSP']:.1f} mm WG</b></td></tr>
    <tr><td>Total Press.</td>     <td style="text-align:right"><b>{_ftp_req:.1f} mm WG</b></td></tr>
    <tr><td>BKW</td>              <td style="text-align:right"><b>{sc['BKW']:.3f} kW</b></td></tr>
    <tr><td>Motor Rating</td>     <td style="text-align:right;color:#00897B"><b>{m_rat['rating_str']}</b></td></tr>
    <tr><td>&eta; Static</td>     <td style="text-align:right"><b>{sc['Static_Eff']:.1f}%</b></td></tr>
    <tr><td>&eta; Total</td>      <td style="text-align:right"><b>{sc['Total_Eff']:.1f}%</b></td></tr>
  </table>
  <hr style="border:none;border-top:1px solid rgba(0,137,123,0.15);margin:.5rem 0">
  <div style="text-align:center;color:{dev_color};font-size:.82rem;font-weight:700">
    {match_lbl} &mdash; &Delta; {dev:.1%} from target
  </div>
</div>""", unsafe_allow_html=True)

        # ── Per-motor expandable detail ───────────────────────────
        st.markdown('')
        st.markdown('#### 📈 Performance Curves per Motor Option')
        for rec in motor_recs:
            m, sc = rec['motor'], rec['scaled']
            m_rat = rec['motor_rating']
            _pfx  = '🏆 ' if rec['recommended'] else ''
            label = f"{_pfx}{m['label']} — Blade {rec['angle']}° — Motor Rating: {m_rat['rating_str']}"
            with st.expander(label, expanded=rec['recommended']):
                flow_val = convert_flow_out(sc['Q_CMH'])
                flow_target = convert_flow_out(req_cmh)
                ic = st.columns(5)
                ic[0].metric('Volume',          f"{flow_val:.0f} {unit}",
                             f"{flow_val-flow_target:+.0f} vs target")
                ic[1].metric('Static Pressure', f"{sc['FSP']:.1f} mm WG",
                             f"{sc['FSP']-req_sp:+.2f} vs target")
                ic[2].metric('BKW',             f"{sc['BKW']:.3f} kW")
                ic[3].metric('Motor Rating',    m_rat['rating_str'], f"Frame {m_rat['frame']}")
                ic[4].metric('η Total',         f"{sc['Total_Eff']:.1f}%")
                st.plotly_chart(
                    create_ml_prediction_curves(predict_performance(df, rec['angle']), df, rec['angle']),
                    use_container_width=True,
                    key=f"motor_curve_{m['rpm']}_{rec['angle']}")

        # ── Comparison table ─────────────────────────────────────
        st.markdown('---')
        st.markdown('#### 📋 Side-by-Side Comparison')
        tbl = []
        for rec in motor_recs:
            m, sc = rec['motor'], rec['scaled']
            m_rat = rec['motor_rating']
            _match_icon = '✅' if rec['deviation'] < 0.3 else '⚠️'
            tbl.append({
                'Motor':               m['label'],
                'Motor Rating':        m_rat['rating_str'],
                'Frame':               m_rat['frame'],
                'Blade Angle (°)':     rec['angle'],
                f'Volume ({unit})':    round(convert_flow_out(sc['Q_CMH'])),
                f'vs Required {unit}': f"{convert_flow_out(sc['Q_CMH'])-convert_flow_out(req_cmh):+.0f}",
                'Outlet V (m/s)':      round(_v_out_req, 2),
                'FTP (mm WG)':         round(_ftp_req, 2),
                'FSP (mm WG)':         round(sc['FSP'], 2),
                'vs Required SP':      f"{sc['FSP']-req_sp:+.2f}",
                'BKW (kW)':            round(sc['BKW'], 3),
                'η Static (%)':        round(sc['Static_Eff'], 1),
                'η Total (%)':         round(sc['Total_Eff'], 1),
                'Match':               f"{_match_icon} {rec['deviation']:.1%}",
                'Best':                '🏆' if rec['recommended'] else '',
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
            f'Volume at BEP ({unit})':  round(convert_flow_out(br['Q_CMH'])),
            'FSP at BEP':     round(br['FSP'], 2),
            'FTP at BEP':     round(br['FTP'], 2),
            'Static Eff (%)': round(br['Static_Eff'], 1),
            'Total Eff (%)':  round(br['Total_Eff'], 1),
            'BKW (kW)':       round(br['BKW'], 3),
        })
    st.dataframe(pd.DataFrame(bep_rows), use_container_width=True, hide_index=True)


# ── FOOTER ─────────────────────────────────────────────────────
st.markdown('---')
st.markdown(
    '<div style="text-align:center;color:#537775;font-size:.82rem;font-weight:500;padding:1rem 0">'
    '🌀 Tube Axial Fan Performance Tool — Engineering Analysis & Performance Modeling'
    '</div>', unsafe_allow_html=True)
