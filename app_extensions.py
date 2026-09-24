"""
app_extensions.py — Sidebar extension pages for the Axial Fan Performance Tool.

Adds three new modes selectable from the sidebar:
  1. 🗄️  Database Manager  — add / edit / delete fans and their test rows
  2. 💾  Model Store       — inspect, force-retrain, and manage saved models
  3. 🌐  Cross-Fan Select  — compare ALL fans and recommend the best match

HOW TO INTEGRATE INTO app.py
──────────────────────────────
Only three edits to the existing app.py are needed:

  (A) At the very top, after the existing imports:

        from app_extensions import render_sidebar_mode_selector, render_extension_page
        from fan_db import init_db
        init_db()   # create tables + seed built-ins once

  (B) Replace the first line inside `with st.sidebar:` with:

        with st.sidebar:
            mode = render_sidebar_mode_selector()
            if mode != "⚙️  Fan Analysis":
                render_extension_page(mode)
                st.stop()
            # ── rest of existing sidebar unchanged ───────────────────

  (C) Swap the existing model-training call:

        # OLD:  mi = _train(selected_fan, ct, df_json)
        # NEW: (No training needed for physics model!)
        # Just use df directly.

  That's it.  No other changes to app.py.

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import functools
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

# ── Company Logo & Header Utilities ───────────────────────────
@functools.lru_cache(maxsize=1)
def get_logo_base64() -> str:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'logo.png')
    if os.path.exists(logo_path):
        try:
            with open(logo_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return ""
    return ""

def render_company_header_html(
    title: str,
    subtitle: str = "ML-Powered Performance Prediction & Engineering Visualisation Tool",
    badge: str = "MAXIM AIR • FAN ENGINEERING SUITE",
) -> str:
    b64 = get_logo_base64()
    img_tag = (
        f'<img src="data:image/png;base64,{b64}" class="companyhead-logo-img" alt="Maxim Air Logo" />'
        if b64 else '<div class="companyhead-logo-fallback">🌀</div>'
    )
    return f"""
    <div class="companyhead-banner">
      <div class="companyhead-logo-container">
        {img_tag}
      </div>
      <div class="companyhead-content">
        <div class="companyhead-badge">{badge}</div>
        <h1 class="companyhead-title">{title}</h1>
        <p class="companyhead-subtitle">{subtitle}</p>
      </div>
    </div>
    """

def render_sidebar_brand_html() -> str:
    b64 = get_logo_base64()
    img_tag = (
        f'<img src="data:image/png;base64,{b64}" class="sidebar-brand-logo" alt="Maxim Air Logo" />'
        if b64 else '<div class="sidebar-brand-logo-fallback">🌀</div>'
    )
    return f"""
    <div class="sidebar-brand-container">
      {img_tag}
      <div class="sidebar-brand-text">
        <div class="sidebar-brand-title">MAXIMAIR</div>
        <div class="sidebar-brand-subtitle">Axial Fan Engineering Suite</div>
      </div>
    </div>
    """

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

# lazy imports of project modules (avoids circular issues at top-level)


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS injected once
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
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

/* Headers & Company Head */
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

/* Sidebar Brand Header */
.sidebar-brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.4rem 0.2rem 0.6rem 0.2rem;
    margin-bottom: 0.2rem;
}
.sidebar-brand-logo {
    width: 46px;
    height: 46px;
    object-fit: contain;
    flex-shrink: 0;
    filter: drop-shadow(0 2px 6px rgba(0, 137, 123, 0.3));
}
.sidebar-brand-logo-fallback {
    font-size: 1.8rem;
}
.sidebar-brand-text {
    display: flex;
    flex-direction: column;
}
.sidebar-brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0F2A28;
    letter-spacing: 0.6px;
    line-height: 1.1;
}
.sidebar-brand-subtitle {
    font-size: 0.70rem;
    font-weight: 600;
    color: #00897B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

.ext-header {
    background: linear-gradient(135deg, #0F2A28 0%, #00897B 100%);
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(0, 137, 123, 0.2);
    box-shadow: 0 4px 18px rgba(0, 137, 123, 0.12);
}
.ext-header h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}
.ext-header p {
    color: rgba(255, 255, 255, 0.9) !important;
    margin: 0.3rem 0 0 0;
    font-size: 0.9rem;
}
.fan-card {
    background: #FFFFFF;
    border: 1px solid #D1E7E5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(15, 42, 40, 0.04);
}
.model-badge-fresh  { color: #00897B; font-weight: 700; }
.model-badge-stale  { color: #D97706; font-weight: 700; }
.model-badge-absent { color: #DC2626; font-weight: 700; }
.info-badge {
    background: #E0F2F1;
    border: 1px solid #80CBC4;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #0F2A28;
    font-size: 0.86rem;
    font-weight: 500;
    margin-bottom: 0.8rem;
}
</style>
"""


def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar mode selector  (call this FIRST inside `with st.sidebar:`)
# ─────────────────────────────────────────────────────────────────────────────

MODES = [
    "🌐  Cross-Fan Selection",
    "⚙️  Fan Analysis",
    "🗄️  Database Manager",
]


def render_sidebar_mode_selector() -> str:
    """
    Renders the top of the sidebar: app logo + mode radio.
    Returns the selected mode string.
    """
    st.markdown(render_sidebar_brand_html(), unsafe_allow_html=True)
    st.markdown("---")
    st.radio("**Flow Volume Unit**", ["CMH", "CFM"], index=0, horizontal=True, key="flow_unit")
    st.markdown("---")
    mode = st.radio("**Mode**", MODES, key="app_mode")
    st.markdown("---")
    return mode


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def render_extension_page(mode: str) -> None:
    _inject_css()
    if mode == "🌐  Cross-Fan Selection":
        _page_cross_fan_selection()
    elif mode == "🗄️  Database Manager":
        _page_db_manager()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across pages
# ─────────────────────────────────────────────────────────────────────────────

def _fan_id_from_name(display_name: str) -> str:
    """Derive a safe fan_id from a display name."""
    return (
        display_name.lower()
        .replace('"', "in")
        .replace("'", "")
        .replace(" ", "_")
        .strip("_")
    )


def _load_all_computed() -> dict[str, pd.DataFrame]:
    """Return {fan_id: computed_df} for every fan in the DB that has rows."""
    from fan_db import list_fans, get_raw_df, get_fan_constants
    from data import compute_derived_quantities

    out = {}
    for fan in list_fans():
        fid = fan["fan_id"]
        try:
            raw = get_raw_df(fid)
            constants = get_fan_constants(fid)
            computed = compute_derived_quantities(df=raw, constants=constants)
            out[fid] = computed
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Database Manager
# ─────────────────────────────────────────────────────────────────────────────

def _page_db_manager() -> None:
    from fan_db import (
        list_fans, get_raw_df, save_raw_df, save_constants,
        get_fan_constants, create_fan, delete_fan, RAW_COLS,
    )

    st.markdown(
        render_company_header_html(
            title="Fan Database Manager",
            subtitle="Add, edit, and manage the cumulative fan test database. Changes update instantly across all tools.",
            badge="MAXIM AIR • DATABASE OPERATIONS",
        ),
        unsafe_allow_html=True,
    )

    # ── top-level action selector ─────────────────────────────────────────────
    action = st.radio(
        "Action",
        ["📋 View / Edit existing fan", "➕ Add new fan", "🗑️ Delete a fan"],
        horizontal=True,
        key="db_action",
    )

    fans = list_fans()
    fan_names = [f["display_name"] for f in fans]
    fan_id_map = {f["display_name"]: f["fan_id"] for f in fans}

    # ──────────────────────────────────────────────────────────────────────────
    # VIEW / EDIT
    # ──────────────────────────────────────────────────────────────────────────
    if action == "📋 View / Edit existing fan":
        if not fans:
            st.info("No fans in the database yet. Use 'Add new fan' to get started.")
            return

        selected_display = st.selectbox("Select fan", fan_names, key="db_edit_fan")
        fan_id = fan_id_map[selected_display]

        st.markdown(
            f'<div class="info-badge">'
            f'Fan ID: <strong>{fan_id}</strong> &nbsp;·&nbsp; '
            f'<span class="model-badge-fresh">✅ Active &amp; Ready</span></div>',
            unsafe_allow_html=True,
        )

        # ── Constants editor ──────────────────────────────────────────────────
        with st.expander("📐 Edit Engineering Constants", expanded=False):
            constants = get_fan_constants(fan_id)
            c1, c2, c3 = st.columns(3)
            new_const = {
                "duct_dia_m":        c1.number_input("Duct Dia (m)",      value=float(constants["duct_dia_m"]),        format="%.4f", key=f"dd_{fan_id}"),
                "discharge_coeff":   c1.number_input("Discharge Coeff",   value=float(constants["discharge_coeff"]),   format="%.2f", key=f"cd_{fan_id}"),
                "cw":                c1.number_input("Wattmeter CW",      value=float(constants["cw"]),                format="%.1f", key=f"cw_{fan_id}"),
                "test_temp_c":       c2.number_input("Test Temp (°C)",    value=float(constants["test_temp_c"]),       step=1.0,      key=f"tt_{fan_id}"),
                "test_baro_mmhg":    c2.number_input("Test Baro (mmHg)", value=float(constants["test_baro_mmhg"]),    step=1.0,      key=f"tb_{fan_id}"),
                "design_temp_c":     c2.number_input("Design Temp (°C)", value=float(constants["design_temp_c"]),     step=1.0,      key=f"dt_{fan_id}"),
                "design_baro_mmhg":  c3.number_input("Design Baro",      value=float(constants["design_baro_mmhg"]), step=1.0,      key=f"db_{fan_id}"),
                "design_speed_rpm":  c3.number_input("Design RPM",       value=float(constants["design_speed_rpm"]), step=1.0,      key=f"ds_{fan_id}"),
                "motor_efficiency":  c3.number_input("Motor Eff",        value=float(constants["motor_efficiency"]),  format="%.2f", key=f"me_{fan_id}"),
                "g": 9.81,
            }
            if st.button("💾 Save Constants", key=f"save_const_{fan_id}"):
                save_constants(fan_id, new_const)
                st.success("Constants saved successfully.")
                st.rerun()

        # ── Raw data editor ───────────────────────────────────────────────────
        st.markdown("### ✏️ Test Data Editor")
        st.markdown(
            '<div class="info-badge">Edit cells, add rows with the ＋ button, '
            "or delete rows with the trash icon. "
            "Click <strong>Save to Database</strong> when done.</div>",
            unsafe_allow_html=True,
        )

        raw_df = get_raw_df(fan_id)
        edited = st.data_editor(
            raw_df,
            num_rows="dynamic",
            use_container_width=True,
            height=380,
            key=f"db_editor_{fan_id}",
            column_config={
                "Srno":  st.column_config.NumberColumn("Sr#",   min_value=1),
                "ANGLE": st.column_config.NumberColumn("Angle°", min_value=0, max_value=90),
                "DEL_P": st.column_config.NumberColumn("ΔP",     format="%.2f"),
                "SP":    st.column_config.NumberColumn("SP",     format="%.2f"),
                "W1":    st.column_config.NumberColumn("W1"),
                "W2":    st.column_config.NumberColumn("W2"),
                "Volt":  st.column_config.NumberColumn("Volt",   format="%.1f"),
                "Amp":   st.column_config.NumberColumn("Amp",    format="%.3f"),
                "RPM":   st.column_config.NumberColumn("RPM",    format="%.0f"),
            },
        )

        col_save, col_reset, _ = st.columns([1, 1, 3])

        if col_save.button("💾 Save to Database", type="primary",
                            use_container_width=True, key=f"save_rows_{fan_id}"):
            clean = edited.dropna(subset=["ANGLE", "DEL_P", "SP"]).reset_index(drop=True)
            new_hash = save_raw_df(fan_id, clean)
            delete_model(fan_id)      # data changed → model stale
            st.success(f"✅ {len(clean)} rows saved. Hash: {new_hash[:12]}…  Model marked stale.")
            st.rerun()

        if col_reset.button("↩️ Reload from DB", use_container_width=True,
                             key=f"reload_rows_{fan_id}"):
            st.rerun()

        # ── Quick summary ─────────────────────────────────────────────────────
        with st.expander("📊 Quick Data Summary"):
            st.dataframe(
                raw_df.describe().round(3),
                use_container_width=True,
            )
            st.markdown(
                f"**{len(raw_df)} rows** across "
                f"**{raw_df['ANGLE'].nunique()} blade angles**: "
                + ", ".join(f"{a}°" for a in sorted(raw_df["ANGLE"].unique()))
            )

    # ──────────────────────────────────────────────────────────────────────────
    # ADD NEW FAN
    # ──────────────────────────────────────────────────────────────────────────
    elif action == "➕ Add new fan":
        st.markdown("### ➕ Register a New Fan")

        with st.form("new_fan_form"):
            st.markdown("**Fan Identity**")
            fc1, fc2 = st.columns(2)
            display_name = fc1.text_input("Display Name", placeholder='e.g. 30" Tube Axial Fan')
            fan_id_input = fc2.text_input(
                "Fan ID (auto or custom)",
                placeholder="e.g. 30in_TA",
                help="Leave blank to auto-generate from display name.",
            )

            st.markdown("**Engineering Constants**")
            cc1, cc2, cc3 = st.columns(3)
            new_fan_const = {
                "duct_dia_m":       cc1.number_input("Duct Dia (m)",     value=0.4572, format="%.4f"),
                "discharge_coeff":  cc1.number_input("Discharge Coeff",  value=0.98,   format="%.2f"),
                "cw":               cc1.number_input("Wattmeter CW",     value=10.0,   format="%.1f"),
                "test_temp_c":      cc2.number_input("Test Temp (°C)",   value=30.0,   step=1.0),
                "test_baro_mmhg":   cc2.number_input("Test Baro (mmHg)", value=760.0,  step=1.0),
                "design_temp_c":    cc3.number_input("Design Temp (°C)", value=30.0,   step=1.0),
                "design_baro_mmhg": cc3.number_input("Design Baro",      value=760.0,  step=1.0),
                "design_speed_rpm": cc3.number_input("Design RPM",       value=1460.0, step=1.0),
                "motor_efficiency": cc3.number_input("Motor Eff",        value=0.81,   format="%.2f"),
                "g": 9.81,
            }

            st.markdown("**Seed Data (optional)** — paste CSV or leave blank to start empty")
            csv_text = st.text_area(
                "CSV data (header: Srno,ANGLE,DEL_P,SP,W1,W2,Volt,Amp,RPM)",
                height=120,
                placeholder="1,20,8.0,7.5,45,5,427,1.33,1459\n2,20,6.0,...",
            )

            submitted = st.form_submit_button("➕ Create Fan", type="primary")

        if submitted:
            if not display_name.strip():
                st.error("Display name is required.")
            else:
                fid = fan_id_input.strip() or _fan_id_from_name(display_name)
                if any(f["fan_id"] == fid for f in fans):
                    st.error(f"Fan ID '{fid}' already exists. Choose a different one.")
                else:
                    seed_df = None
                    if csv_text.strip():
                        import io
                        try:
                            seed_df = pd.read_csv(io.StringIO(csv_text.strip()))
                        except Exception as e:
                            st.error(f"CSV parse error: {e}")
                            return
                    create_fan(fid, display_name.strip(), new_fan_const, seed_df)
                    st.success(f"✅ Fan '{display_name}' created with ID '{fid}'.")
                    st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # DELETE FAN
    # ──────────────────────────────────────────────────────────────────────────
    elif action == "🗑️ Delete a fan":
        st.markdown("### 🗑️ Remove a Fan from the Database")
        st.warning("⚠️ This permanently deletes the fan, all its test rows, and its saved configurations.")

        if not fans:
            st.info("No fans in the database.")
            return

        del_name = st.selectbox("Fan to delete", fan_names, key="db_del_fan")
        del_id = fan_id_map[del_name]
        confirm = st.text_input(
            f'Type **{del_name}** to confirm deletion', key="del_confirm"
        )

        if st.button("🗑️ Delete permanently", type="primary", key="do_delete"):
            if confirm.strip() == del_name:
                delete_fan(del_id)
                st.success(f"Fan '{del_name}' deleted.")
                st.rerun()
            else:
                st.error("Name does not match. Deletion cancelled.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Cross-Fan Selection
# ─────────────────────────────────────────────────────────────────────────────

def _page_cross_fan_selection() -> None:
    from fan_db import list_fans, get_fan_constants
    from physics_model import cross_fan_recommend, select_motor_rating

    # ── Inject page-specific CSS ──────────────────────────────────────────────
    st.markdown("""
    <style>
    .cfs-hero {
        background: linear-gradient(135deg, #0F2A28 0%, #00897B 100%);
        padding: 1.3rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0, 137, 123, 0.25);
        box-shadow: 0 6px 24px rgba(0, 137, 123, 0.15);
    }
    .cfs-hero h2 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .cfs-hero p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.25rem 0 0 0;
        font-size: 0.88rem;
        line-height: 1.4;
    }

    .cfs-section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F2A28;
        margin: 1.4rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0, 137, 123, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .cfs-best-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #E0F2F1 100%);
        border: 2px solid #00897B;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0, 137, 123, 0.12);
    }
    .cfs-best-card::after {
        content: '🏆';
        position: absolute;
        top: 12px;
        right: 16px;
        font-size: 2.5rem;
        opacity: 0.15;
    }
    .cfs-best-card .best-badge {
        display: inline-block;
        background: #00897B;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 0.6rem;
    }
    .cfs-best-card .fan-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F2A28;
        margin: 0.3rem 0;
    }
    .cfs-best-card .fan-config {
        font-size: 1rem;
        color: #3B5957;
        margin-bottom: 0.8rem;
    }
    .cfs-best-card .specs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .cfs-best-card .spec-item {
        text-align: center;
        padding: 0.65rem 0.5rem;
        background: #FFFFFF;
        border: 1px solid rgba(0, 137, 123, 0.2);
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(15, 42, 40, 0.04);
    }
    .cfs-best-card .spec-item .spec-val {
        font-size: 1.2rem;
        font-weight: 800;
        color: #00897B;
    }
    .cfs-best-card .spec-item .spec-label {
        font-size: 0.7rem;
        color: #537775;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
        font-weight: 600;
    }

    .cfs-runner-card {
        background: #FFFFFF;
        border: 1.5px solid rgba(0, 137, 123, 0.22);
        border-radius: 14px;
        padding: 1.4rem;
        height: 100%;
        box-shadow: 0 2px 12px rgba(15, 42, 40, 0.05);
    }
    .cfs-runner-card .rank-badge {
        display: inline-block;
        background: #E0F2F1;
        color: #00897B;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 137, 123, 0.3);
        margin-bottom: 0.5rem;
    }
    .cfs-runner-card .fan-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F2A28;
        margin: 0.3rem 0;
    }
    .cfs-runner-card .motor-info {
        font-size: 0.85rem;
        color: #537775;
        margin-bottom: 0.6rem;
        font-weight: 500;
    }
    .cfs-runner-card table {
        width: 100%;
        font-size: 0.82rem;
        color: #2C3E3D;
    }
    .cfs-runner-card table td {
        padding: 0.25rem 0;
    }
    .cfs-runner-card table td:last-child {
        text-align: right;
        font-weight: 700;
        color: #0F2A28;
    }
    .cfs-runner-card .deviation-bar {
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(0, 137, 123, 0.15);
        text-align: center;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .cfs-bkw-tag {
        display: inline-block;
        background: #E0F2F1;
        color: #00897B;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 8px;
        border: 1px solid #00897B;
        margin-left: 0.4rem;
    }

    .cfs-divider {
        border: none;
        border-top: 1px solid rgba(0, 137, 123, 0.15);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Header ───────────────────────────────────────────────────────────
    st.markdown(
        render_company_header_html(
            title="Cross-Fan Selection Engine",
            subtitle="Evaluate fans across motor speeds to find the optimal selection with motor recommendations.",
            badge="MAXIM AIR • CROSS-FAN SELECTION",
        ),
        unsafe_allow_html=True,
    )

    fans = list_fans()
    if not fans:
        st.info("No fans in the database.")
        return

    eligible = fans
    all_names = [f["display_name"] for f in eligible]

    # ── Compact Fan Selection ─────────────────────────────────────────────────
    chosen_names = st.multiselect(
        "🌀 Select Fans to Compare (leave empty for all)",
        all_names, default=all_names, key="cfs_fans",
        help="Choose which fan models to include in the search",
    )
    if not chosen_names:
        chosen_names = all_names

    fan_name_to_id = {f["display_name"]: f["fan_id"] for f in eligible}
    chosen_ids = [fan_name_to_id[n] for n in chosen_names]

    # ── System Requirements & Inline BKW Override ─────────────────────────────
    if "cfs_bkw_active" not in st.session_state:
        st.session_state.cfs_bkw_active = False

    def _toggle_cfs_bkw():
        st.session_state.cfs_bkw_active = not st.session_state.cfs_bkw_active

    unit = flow_unit_label()
    rc1, rc2, rc3, rc4 = st.columns([1.1, 1.0, 1.0, 1.5])

    if unit == 'CFM':
        req_val = rc1.number_input(
            "Volume (CFM)",
            int(100 * CMH_TO_CFM), int(500000 * CMH_TO_CFM),
            int(10000 * CMH_TO_CFM), int(100 * CMH_TO_CFM),
            key="cfs_cfm",
        )
        req_cmh = req_val * CFM_TO_CMH
    else:
        req_cmh = rc1.number_input(
            "Volume (CMH)", 100, 500000, 10000, 100, key="cfs_cmh",
        )

    req_sp = rc2.number_input(
        "SP (mm WG)", 0.0, 200.0, 10.0, 0.5, key="cfs_sp",
    )

    allowed_poles = rc3.multiselect(
        "Motor Poles", [2, 4, 6], default=[6], key="cfs_poles",
    )

    # Inline BKW Override Field
    bkw_active = st.session_state.cfs_bkw_active
    rc4.markdown("**Predefined BKW (kW)**")
    bkw_btn_col, bkw_input_col = rc4.columns([0.85, 1.35])

    if bkw_active:
        bkw_btn_col.button(
            "🔴 Disable",
            on_click=_toggle_cfs_bkw,
            key="cfs_bkw_btn",
            help="Click to disable BKW override and use physics-computed BKW",
            use_container_width=True,
        )
        bkw_override_val = bkw_input_col.number_input(
            "BKW Override",
            0.01, 500.0,
            float(st.session_state.get("cfs_bkw_val", 0.75)),
            0.01,
            format="%.3f",
            key="cfs_bkw_val",
            label_visibility="collapsed",
            help="Custom motor BKW. Efficiencies will be back-calculated: η = AirPower / BKW × 100",
        )
    else:
        bkw_override_val = None
        bkw_btn_col.button(
            "⚡ Use",
            on_click=_toggle_cfs_bkw,
            key="cfs_bkw_btn",
            help="Click to enter a custom BKW to override physics calculations",
            use_container_width=True,
        )
        bkw_input_col.text_input(
            "BKW Override",
            value="Auto (Computed)",
            disabled=True,
            label_visibility="collapsed",
            help="Click '⚡ Use' to enter custom BKW",
        )

    # ── Run Button ────────────────────────────────────────────────────────────
    st.markdown("")
    run_btn = st.button(
        "🔍  Find Best Fan–Motor–Angle Combination",
        type="primary", use_container_width=True, key="cfs_run",
    )

    if not run_btn:
        # Show a subtle prompt
        st.markdown(
            '<div style="text-align:center;color:#3B5957;'
            'font-size:0.95rem;font-weight:500;padding:2.2rem 1rem;'
            'background:#F8FAF9;border-radius:12px;border:1.5px dashed #B2DFDB;'
            'margin-top:1.2rem">'
            '👆 Configure your system requirements above and click <strong>Find Best Fan–Motor–Angle Combination</strong> to evaluate options'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Load computed DataFrames ──────────────────────────────────────────────
    with st.spinner("🔄 Evaluating all fans across motor speeds …"):
        computed_map = _load_all_computed()
        computed_map = {k: v for k, v in computed_map.items() if k in chosen_ids}

        if not computed_map:
            st.error("Could not load computed data for the selected fans.")
            return

        recommendations = cross_fan_recommend(
            chosen_ids, computed_map, req_cmh, req_sp, allowed_poles,
        )

    if not recommendations:
        st.error("No recommendations could be generated. Check fan data.")
        return

    # ── Helper: apply BKW override to a scaled dict ───────────────────────────
    def _apply_bkw_override(sc: dict, bkw_val: float, q_cmh: float) -> dict:
        """Recalculate efficiencies using a user-supplied BKW."""
        out = dict(sc)
        out['BKW'] = bkw_val

        # FTP and FSP remain physics-derived (they depend on fan geometry, not motor)
        ftp = out.get('FTP', 0)
        fsp = out.get('FSP', 0)

        # Air Power Total = 2.725 × Q × FTP × 10⁻⁶
        air_power_t = 2.725 * q_cmh * ftp * 1e-6
        # Air Power Static = 2.725 × Q × max(FSP, 0) × 10⁻⁶
        air_power_st = 2.725 * q_cmh * max(fsp, 0) * 1e-6

        out['Total_Eff']  = (air_power_t / bkw_val * 100) if bkw_val > 0 else 0
        out['Static_Eff'] = (air_power_st / bkw_val * 100) if bkw_val > 0 else 0
        out['Total_Eff']  = max(0, min(out['Total_Eff'], 100))
        out['Static_Eff'] = max(0, min(out['Static_Eff'], 100))
        return out

    # ── Apply BKW override if active ──────────────────────────────────────────
    if bkw_override_val is not None:
        for rec in recommendations:
            rec['scaled'] = _apply_bkw_override(
                rec['scaled'], bkw_override_val, rec['scaled']['Q_CMH']
            )
        # Re-sort by deviation (unchanged) — deviation is geometric, not BKW-dependent
        recommendations.sort(key=lambda r: r['deviation'])
        for i, r in enumerate(recommendations):
            r['recommended'] = (i == 0)

    # Ensure motor_rating is fresh with the active BKW for each recommendation
    for rec in recommendations:
        rec['motor_rating'] = select_motor_rating(
            rec['scaled']['BKW'], rec.get('poles', 6)
        )

    bkw_tag = (' <span class="cfs-bkw-tag">BKW Override Active</span>'
               if bkw_override_val is not None else '')

    # ── Results Section ───────────────────────────────────────────────────────
    st.markdown('<hr class="cfs-divider">', unsafe_allow_html=True)

    # ── 🏆 Best Match — Hero Card ─────────────────────────────────────────────
    best = recommendations[0]
    best_sc = best['scaled']
    best_m_rat = best['motor_rating']
    st.markdown(f"""
    <div class="cfs-best-card">
      <div class="best-badge">🏆 Best Match</div>
      <div class="fan-name">{best['fan_name']}{bkw_tag}</div>
      <div class="fan-config">
        {best['motor_label']} &nbsp;·&nbsp; Blade {best['angle']}°
        &nbsp;·&nbsp; Motor Rating: <b>{best_m_rat['rating_str']}</b> (BKW+20%)
        &nbsp;·&nbsp; Δ {best['deviation']:.1%} from target
      </div>
      <div class="specs-grid">
        <div class="spec-item">
          <div class="spec-val">{convert_flow_out(best_sc['Q_CMH']):.0f}</div>
          <div class="spec-label">{unit}</div>
        </div>
        <div class="spec-item">
          <div class="spec-val">{best_sc['FSP']:.1f}</div>
          <div class="spec-label">FSP mm WG</div>
        </div>
        <div class="spec-item">
          <div class="spec-val">{best_sc['FTP']:.1f}</div>
          <div class="spec-label">FTP mm WG</div>
        </div>
        <div class="spec-item">
          <div class="spec-val">{best_sc['BKW']:.3f}</div>
          <div class="spec-label">BKW kW</div>
        </div>
        <div class="spec-item">
          <div class="spec-val" style="color:#00897B;font-weight:800">{best_m_rat['rating_short']}</div>
          <div class="spec-label">Motor Rating</div>
        </div>
        <div class="spec-item">
          <div class="spec-val">{best_sc['Static_Eff']:.1f}%</div>
          <div class="spec-label">η Static</div>
        </div>
        <div class="spec-item">
          <div class="spec-val">{best_sc['Total_Eff']:.1f}%</div>
          <div class="spec-label">η Total</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Runner-up cards (ranks 2 & 3) ─────────────────────────────────────────
    runners = recommendations[1:4]
    if runners:
        st.markdown(
            '<div class="cfs-section-title"><span class="icon">🥈</span> Runner-Up Options</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(runners))
        for idx, rec in enumerate(runners):
            sc = rec['scaled']
            m_rat = rec['motor_rating']
            dev = rec['deviation']
            dev_color = '#059669' if dev < 0.2 else ('#D97706' if dev < 0.5 else '#DC2626')
            match_lbl = 'Excellent' if dev < 0.2 else ('Good' if dev < 0.4 else ('Fair' if dev < 0.7 else 'Poor'))

            with cols[idx]:
                st.markdown(f"""
                <div class="cfs-runner-card">
                  <div class="rank-badge">#{idx + 2}</div>
                  <div class="fan-name">{rec['fan_name']}</div>
                  <div class="motor-info">{rec['motor_label']} · {rec['angle']}°</div>
                  <table>
                    <tr><td>{unit}</td><td>{convert_flow_out(sc['Q_CMH']):.0f}</td></tr>
                    <tr><td>FSP</td><td>{sc['FSP']:.1f} mm WG</td></tr>
                    <tr><td>FTP</td><td>{sc['FTP']:.1f} mm WG</td></tr>
                    <tr><td>BKW</td><td>{sc['BKW']:.3f} kW</td></tr>
                    <tr><td>Motor Rating</td><td style="color:#00897B"><b>{m_rat['rating_str']}</b></td></tr>
                    <tr><td>η Static</td><td>{sc['Static_Eff']:.1f}%</td></tr>
                    <tr><td>η Total</td><td>{sc['Total_Eff']:.1f}%</td></tr>
                  </table>
                  <div class="deviation-bar" style="color:{dev_color}">
                    {match_lbl} — Δ {dev:.1%}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Complete Ranked Table ──────────────────────────────────────────────────
    st.markdown('<hr class="cfs-divider">', unsafe_allow_html=True)
    bkw_label = " (BKW Override)" if bkw_override_val is not None else ""
    st.markdown(
        f'<div class="cfs-section-title"><span class="icon">📋</span>'
        f' All Combinations Ranked{bkw_label}</div>',
        unsafe_allow_html=True,
    )

    _tbl_ranked = []
    for _i, _rec in enumerate(recommendations):
        _sc = _rec["scaled"]
        _m_rat = _rec["motor_rating"]
        _di = "✅" if _rec["deviation"] < 0.3 else ("⚠️" if _rec["deviation"] < 0.6 else "❌")
        row_data = {
            "Rank":             _i + 1,
            "Fan":              _rec["fan_name"],
            "Motor":            _rec["motor_label"],
            "Motor Rating":     _m_rat["rating_str"],
            "Angle (°)":        _rec["angle"],
            f"Volume ({unit})": round(convert_flow_out(_sc["Q_CMH"])),
            f"Δ {unit}":        f"{convert_flow_out(_sc['Q_CMH']) - convert_flow_out(req_cmh):+.0f}",
            "FSP (mm WG)":      round(_sc["FSP"], 2),
            "Δ SP":             f"{_sc['FSP'] - req_sp:+.2f}",
            "BKW (kW)":         round(_sc["BKW"], 3),
            "η Static (%)":     round(_sc["Static_Eff"], 1),
            "η Total (%)":      round(_sc["Total_Eff"], 1),
            "Match":            f"{_di} {_rec['deviation']:.1%}",
        }
        _tbl_ranked.append(row_data)

    _tbl_ranked_df = pd.DataFrame(_tbl_ranked)
    st.dataframe(
        _tbl_ranked_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 38 + 35 * len(_tbl_ranked)),
    )

    dl_c1, dl_c2, _ = st.columns([1, 1, 3])
    dl_c1.download_button(
        "📥 Download CSV",
        _tbl_ranked_df.to_csv(index=False),
        "cross_fan_selection.csv",
        "text/csv",
        key="cfs_download_ranked",
    )

    # ── Nominal Speed Performance Table ───────────────────────────────────────
    st.markdown('<hr class="cfs-divider">', unsafe_allow_html=True)
    disp_flow = convert_flow_out(req_cmh)
    st.markdown(
        f'<div class="cfs-section-title"><span class="icon">📈</span>'
        f' Nominal Speed Performance @ {disp_flow:.0f} {unit}{bkw_label}</div>',
        unsafe_allow_html=True,
    )

    from physics_model import predict_performance as _pfan
    from scipy.interpolate import interp1d as _i1d

    def _interp_at(pred_df, cmh, col):
        ps = pred_df.sort_values("Q_CMH")
        try:
            val = float(_i1d(
                ps["Q_CMH"], ps[col],
                kind="linear", fill_value="extrapolate",
            )(cmh))
            if np.isnan(val) or np.isinf(val):
                return np.nan
            if col in ["Static_Eff", "Total_Eff"]:
                fsp_val = float(_i1d(
                    ps["Q_CMH"], ps["FSP"],
                    kind="linear", fill_value="extrapolate",
                )(cmh))
                if fsp_val <= 0 and col == "Static_Eff":
                    return 0.0
                return max(0.0, val)
            return val
        except Exception:
            return np.nan

    op_rows = []
    for rec in recommendations:
        fid = rec["fan_id"]
        computed = computed_map.get(fid)
        if computed is None:
            continue
        try:
            pf = _pfan(computed, rec["angle"])
            fsp_val = float(_interp_at(pf, req_cmh, "FSP"))
            ftp_val = float(_interp_at(pf, req_cmh, "FTP"))
            bkw_val = float(_interp_at(pf, req_cmh, "BKW"))
            seff_val = float(_interp_at(pf, req_cmh, "Static_Eff"))
            teff_val = float(_interp_at(pf, req_cmh, "Total_Eff"))

            # Apply BKW override to nominal-speed table too
            if bkw_override_val is not None:
                bkw_val = bkw_override_val
                air_power_t = 2.725 * req_cmh * ftp_val * 1e-6
                air_power_st = 2.725 * req_cmh * max(fsp_val, 0) * 1e-6
                teff_val = (air_power_t / bkw_val * 100) if bkw_val > 0 else 0
                seff_val = (air_power_st / bkw_val * 100) if bkw_val > 0 else 0
                teff_val = max(0, min(teff_val, 100))
                seff_val = max(0, min(seff_val, 100))

            nom_m_rat = select_motor_rating(bkw_val, rec.get("poles", 6))

            op_rows.append({
                "Fan":          str(rec["fan_name"]),
                "Motor":        str(rec["motor_label"]),
                "Motor Rating": nom_m_rat["rating_str"],
                "Angle (°)":    float(rec["angle"]),
                "FSP (mm WG)":  round(fsp_val, 2),
                "FTP (mm WG)":  round(ftp_val, 2),
                "BKW (kW)":     round(bkw_val, 3),
                "η Static (%)": round(seff_val, 1),
                "η Total (%)":  round(teff_val, 1),
            })
        except Exception:
            pass

    if op_rows:
        st.dataframe(
            pd.DataFrame(op_rows),
            use_container_width=True,
            hide_index=True,
            height=min(400, 38 + 35 * len(op_rows)),
        )
    else:
        st.info("Could not compute nominal-speed estimates.")

    # ── Full Predicted Curves — Top Matching Models ───────────────────────────
    from plots import create_ml_prediction_curves
    from physics_model import predict_performance

    top_recs = [r for r in recommendations if r["deviation"] < 0.6]
    if top_recs:
        st.markdown('<hr class="cfs-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="cfs-section-title"><span class="icon">📈</span>'
            ' Performance Curves — Top Matches</div>',
            unsafe_allow_html=True,
        )

        for rec in top_recs:
            fid = rec["fan_id"]
            computed = computed_map.get(fid)
            _pfx = "🏆 " if rec["recommended"] else ""
            label = f"{_pfx}{rec['fan_name']} — {rec['motor_label']} — {rec['angle']}°"
            with st.expander(label, expanded=rec["recommended"]):
                # Summary metrics row
                sc = rec['scaled']
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Volume", f"{convert_flow_out(sc['Q_CMH']):.0f} {unit}")
                mc2.metric("FSP", f"{sc['FSP']:.1f} mm WG")
                mc3.metric("BKW", f"{sc['BKW']:.3f} kW")
                mc4.metric("η Total", f"{sc['Total_Eff']:.1f}%")

                try:
                    pred_df = predict_performance(computed, rec["angle"])
                    fig = create_ml_prediction_curves(pred_df, computed, rec["angle"])
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"cfs_curve_{fid}_{rec['angle']}_{rec['motor_label']}",
                    )
                except Exception as e:
                    st.warning(f"Could not render curve: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;color:#537775;'
        'font-size:0.8rem;padding:2rem 0;font-weight:500">'
        '🌀 Cross-Fan Selection · Physics-based interpolation across all catalogue options'
        '</div>',
        unsafe_allow_html=True,
    )
