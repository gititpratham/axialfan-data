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
        # NEW:
        from model_store import get_or_train_model
        mi = get_or_train_model(
            fan_id=_fan_id_from_name(selected_fan),
            df_computed=df,
            force_retrain=False,
        )

  That's it.  No other changes to app.py.

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

# lazy imports of project modules (avoids circular issues at top-level)


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS injected once
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.ext-header {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    padding:1.6rem 2rem; border-radius:14px; margin-bottom:1.2rem;
    border:1px solid rgba(255,255,255,.1);
}
.ext-header h2 { margin:0; font-size:1.5rem; font-weight:700;
    background:linear-gradient(90deg,#00D4FF,#FF6BFF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.ext-header p { color:rgba(255,255,255,.55); margin:.4rem 0 0 0; font-size:.9rem; }
.fan-card {
    background:rgba(15,12,41,.7); border:1px solid rgba(255,255,255,.1);
    border-radius:12px; padding:1rem 1.2rem; margin-bottom:.8rem;
}
.model-badge-fresh  { color:#00FF85; font-weight:600; }
.model-badge-stale  { color:#FFD700; font-weight:600; }
.model-badge-absent { color:#FF4444; font-weight:600; }
.info-badge {
    background:rgba(0,212,255,.12); border:1px solid rgba(0,212,255,.28);
    border-radius:8px; padding:.7rem 1rem; color:#00D4FF;
    font-size:.86rem; margin-bottom:.8rem;
}
</style>
"""


def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar mode selector  (call this FIRST inside `with st.sidebar:`)
# ─────────────────────────────────────────────────────────────────────────────

MODES = [
    "⚙️  Fan Analysis",
    "🗄️  Database Manager",
    "💾  Model Store",
    "🌐  Cross-Fan Selection",
]


def render_sidebar_mode_selector() -> str:
    """
    Renders the top of the sidebar: app logo + mode radio.
    Returns the selected mode string.
    """
    st.markdown("## 🌀 Axial Fan Tool")
    st.markdown("---")
    mode = st.radio("**Mode**", MODES, key="app_mode")
    st.markdown("---")
    return mode


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def render_extension_page(mode: str) -> None:
    _inject_css()
    if mode == "🗄️  Database Manager":
        _page_db_manager()
    elif mode == "💾  Model Store":
        _page_model_store()
    elif mode == "🌐  Cross-Fan Selection":
        _page_cross_fan_selection()


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
    from model_store import is_model_stale, delete_model

    st.markdown("""
    <div class="ext-header">
      <h2>🗄️ Fan Database Manager</h2>
      <p>Add, edit, and manage the cumulative fan test database. Changes automatically
         mark the ML model as stale until you retrain.</p>
    </div>""", unsafe_allow_html=True)

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
        stale = is_model_stale(fan_id)

        st.markdown(
            f'<div class="info-badge">'
            f'Model status: '
            f'<span class="{"model-badge-stale" if stale else "model-badge-fresh"}">'
            f'{"⚠️ Stale — retrain needed" if stale else "✅ Up to date"}'
            f'</span></div>',
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
                delete_model(fan_id)   # constants changed → model stale
                st.success("Constants saved. Model marked stale.")
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
                "design_temp_c":    cc2.number_input("Design Temp (°C)", value=30.0,   step=1.0),
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
        st.warning("⚠️ This permanently deletes the fan, all its test rows, and its saved model.")

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
                from model_store import delete_model
                delete_model(del_id)
                delete_fan(del_id)
                st.success(f"Fan '{del_name}' deleted.")
                st.rerun()
            else:
                st.error("Name does not match. Deletion cancelled.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Model Store
# ─────────────────────────────────────────────────────────────────────────────

def _page_model_store() -> None:
    from fan_db import list_fans, get_raw_df, get_fan_constants
    from data import compute_derived_quantities
    from model_store import (
        list_stored_models, get_or_train_model,
        is_model_stale, delete_model,
    )

    st.markdown("""
    <div class="ext-header">
      <h2>💾 ML Model Store</h2>
      <p>Inspect saved models, trigger retraining, and manage per-fan model files.
         Models are only retrained when data or constants have changed.</p>
    </div>""", unsafe_allow_html=True)

    fans = list_fans()
    if not fans:
        st.info("No fans in the database.")
        return

    stored = {m["fan_id"]: m for m in list_stored_models()}
    fan_id_map = {f["fan_id"]: f for f in fans}

    # ── Status overview table ─────────────────────────────────────────────────
    st.markdown("### 📊 Model Status Overview")

    rows = []
    for fan in fans:
        fid = fan["fan_id"]
        meta = stored.get(fid)
        stale = is_model_stale(fid)
        rows.append({
            "Fan": fan["display_name"],
            "Model": meta["best_model_name"] if meta else "—",
            "Avg CV R²": f"{meta['avg_r2_cv']:.4f}" if meta else "—",
            "Saved At": meta.get("saved_at", "—") if meta else "—",
            "Status": "⚠️ Stale" if stale else ("✅ Fresh" if meta else "❌ None"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Per-fan train / inspect ───────────────────────────────────────────────
    st.markdown("### 🔧 Per-Fan Model Actions")

    selected_display = st.selectbox(
        "Select fan", [f["display_name"] for f in fans], key="ms_fan"
    )
    fan = next(f for f in fans if f["display_name"] == selected_display)
    fan_id = fan["fan_id"]
    meta = stored.get(fan_id)
    stale = is_model_stale(fan_id)

    status_cls = "model-badge-stale" if stale else ("model-badge-fresh" if meta else "model-badge-absent")
    status_txt = "⚠️ Stale" if stale else ("✅ Fresh" if meta else "❌ No model saved")
    st.markdown(
        f'<div class="info-badge">Status: <span class="{status_cls}">{status_txt}</span></div>',
        unsafe_allow_html=True,
    )

    # Action buttons
    col_train, col_delete, _ = st.columns([1, 1, 3])

    do_train = col_train.button(
        "🔄 Train & Save" if stale or not meta else "♻️ Force Retrain",
        type="primary", use_container_width=True, key=f"train_{fan_id}",
    )
    do_delete = col_delete.button(
        "🗑️ Delete Model", use_container_width=True,
        key=f"del_model_{fan_id}", disabled=meta is None,
    )

    if do_train:
        with st.spinner(f"Training models for {selected_display} …"):
            try:
                raw = get_raw_df(fan_id)
                constants = get_fan_constants(fan_id)
                computed = compute_derived_quantities(df=raw, constants=constants)
                mi = get_or_train_model(fan_id, computed, force_retrain=True)
                st.success(
                    f"✅ Best model: **{mi['best_model_name']}** — "
                    f"Avg CV R² = {mi['results'][mi['best_model_name']]['avg_r2_cv']:.4f}"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    if do_delete:
        delete_model(fan_id)
        st.warning(f"Model for '{selected_display}' deleted.")
        st.rerun()

    # ── Detailed metrics for stored model ────────────────────────────────────
    if meta:
        with st.expander("📋 Stored Model Metadata"):
            st.json(meta)

        with st.expander("📈 Live Model Metrics (load + score)"):
            try:
                from ml_model import TARGET_COLS
                raw = get_raw_df(fan_id)
                constants = get_fan_constants(fan_id)
                computed = compute_derived_quantities(df=raw, constants=constants)
                mi = get_or_train_model(fan_id, computed, force_retrain=False)
                best = mi["best_model_name"]
                res = mi["results"][best]

                mdf = pd.DataFrame({
                    "Target": TARGET_COLS,
                    "Train R²": [res["r2_train"][t] for t in TARGET_COLS],
                    "CV R²":    [res["r2_cv"][t]    for t in TARGET_COLS],
                    "Train RMSE": [res["rmse_train"][t] for t in TARGET_COLS],
                    "CV RMSE":    [res["rmse_cv"][t]    for t in TARGET_COLS],
                })
                st.dataframe(mdf.round(4), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not load metrics: {e}")

    # ── Batch train all stale fans ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Batch Operations")

    stale_fans = [f for f in fans if is_model_stale(f["fan_id"])]
    if stale_fans:
        st.markdown(
            f'<div class="info-badge">⚠️ {len(stale_fans)} fan(s) have stale or missing models: '
            + ", ".join(f["display_name"] for f in stale_fans)
            + "</div>",
            unsafe_allow_html=True,
        )
        if st.button(f"🔄 Train All {len(stale_fans)} Stale Fan(s)", type="primary"):
            prog = st.progress(0, text="Starting …")
            for i, fan in enumerate(stale_fans):
                fid = fan["fan_id"]
                prog.progress((i) / len(stale_fans), text=f"Training {fan['display_name']} …")
                try:
                    raw = get_raw_df(fid)
                    constants = get_fan_constants(fid)
                    computed = compute_derived_quantities(df=raw, constants=constants)
                    get_or_train_model(fid, computed, force_retrain=True)
                except Exception as e:
                    st.warning(f"{fan['display_name']}: {e}")
            prog.progress(1.0, text="Done!")
            st.success("All stale models retrained.")
            st.rerun()
    else:
        st.success("✅ All fans have fresh models.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Cross-Fan Selection
# ─────────────────────────────────────────────────────────────────────────────

def _page_cross_fan_selection() -> None:
    from fan_db import list_fans
    from model_store import cross_fan_recommend, is_model_stale, list_stored_models

    st.markdown("""
    <div class="ext-header">
      <h2>🌐 Cross-Fan Selection</h2>
      <p>Enter your system requirements. Every fan with a saved model is evaluated
         across all standard motor speeds. The best fan–motor–angle combination
         is recommended.</p>
    </div>""", unsafe_allow_html=True)

    fans = list_fans()
    if not fans:
        st.info("No fans in the database.")
        return

    stored_ids = {m["fan_id"] for m in list_stored_models()}
    eligible = [f for f in fans if f["fan_id"] in stored_ids]
    stale_fans = [f for f in fans if is_model_stale(f["fan_id"])]

    if stale_fans:
        st.markdown(
            '<div class="info-badge">⚠️ Some fans have stale or missing models: '
            + ", ".join(f["display_name"] for f in stale_fans)
            + ". Go to <strong>Model Store</strong> to retrain them first.</div>",
            unsafe_allow_html=True,
        )

    if not eligible:
        st.warning("No fans have trained models. Please go to Model Store and train first.")
        return

    # ── Fan multi-select ──────────────────────────────────────────────────────
    st.markdown("### 🌀 Fans to Include")
    all_names = [f["display_name"] for f in eligible]
    chosen_names = st.multiselect(
        "Select fans to compare", all_names, default=all_names, key="cfs_fans",
        placeholder="Leave empty to include all fans",
    )
    if not chosen_names:          # empty = all fans
        chosen_names = all_names

    fan_name_to_id = {f["display_name"]: f["fan_id"] for f in eligible}
    chosen_ids = [fan_name_to_id[n] for n in chosen_names]

    # ── Requirements ─────────────────────────────────────────────────────────
    st.markdown("### 📋 System Requirements")
    rc1, rc2, rc3 = st.columns([1, 1, 1])
    req_cmh = rc1.number_input("Required Volume (CMH)", 100, 500_000, 5000, 100, key="cfs_cmh")
    req_sp  = rc2.number_input("Required SP (mm WG)",   0.0, 200.0,   10.0, 0.5,  key="cfs_sp")
    run_btn = rc3.button("🔍 Find Best Fan", type="primary",
                          use_container_width=True, key="cfs_run")

    if not run_btn:
        return

    # ── Load computed DataFrames ──────────────────────────────────────────────
    with st.spinner("Loading fan data and querying models …"):
        computed_map = _load_all_computed()
        computed_map = {k: v for k, v in computed_map.items() if k in chosen_ids}

        if not computed_map:
            st.error("Could not load computed data for the selected fans.")
            return

        recommendations = cross_fan_recommend(chosen_ids, computed_map, req_cmh, req_sp)

    if not recommendations:
        st.error("No recommendations could be generated. Check that models are trained.")
        return

    # ── BEST MATCH banner ─────────────────────────────────────────────────────
    best = recommendations[0]
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,rgba(0,255,133,.15),rgba(0,212,255,.1));
             border:1px solid rgba(0,255,133,.4); border-radius:14px;
             padding:1.4rem 1.8rem; margin-bottom:1rem;">
          <div style="font-size:.78rem;color:rgba(255,255,255,.5);
               text-transform:uppercase;letter-spacing:1px">🏆 Best Match</div>
          <div style="font-size:2rem;font-weight:700;color:#00FF85;margin:.3rem 0">
            {best['fan_name']}
          </div>
          <div style="font-size:1rem;color:#E0E0E0">
            {best['motor_label']} &nbsp;|&nbsp; Blade angle {best['angle']}°
            &nbsp;|&nbsp; Deviation {best['deviation']:.1%}
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


    # ── ML Operating Point Table — all sensible models ───────────────────────
    st.markdown("---")
    st.markdown(f"### 📈 Performance at Your Operating Point — {req_cmh:.0f} CMH / {req_sp:.1f} mm WG")

    from model_store import predict_for_fan as _pfan
    from scipy.interpolate import interp1d as _i1d

    def _ml_at(pred_df, cmh, col):
        ps = pred_df.sort_values("Q_CMH")
        try:
            return float(_i1d(ps["Q_CMH"], ps[col], kind="linear", fill_value="extrapolate")(cmh))
        except Exception:
            return float("nan")

    op_rows = []
    for rec in recommendations:
        fid     = rec["fan_id"]
        computed = computed_map.get(fid)
        try:
            pf = _pfan(fid, computed, rec["angle"])
            op_rows.append({
                "Fan":           rec["fan_name"],
                "Motor":         rec["motor_label"],
                "Angle (°)":     rec["angle"],
                "FSP (mm WG)":   round(_ml_at(pf, req_cmh, "FSP"), 2),
                "FTP (mm WG)":   round(_ml_at(pf, req_cmh, "FTP"), 2),
                "Power (W)":     round(_ml_at(pf, req_cmh, "BKW") * 1000, 1),
                "η Static (%)":  round(_ml_at(pf, req_cmh, "Static_Eff"), 1),
                "η Total (%)":   round(_ml_at(pf, req_cmh, "Total_Eff"), 1),
            })
        except Exception:
            pass

    if op_rows:
        st.dataframe(pd.DataFrame(op_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Could not compute ML estimates.")

    # ── All Combinations Ranked table ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 All Combinations Ranked")

    _tbl_ranked = []
    for _i, _rec in enumerate(recommendations):
        _sc = _rec["scaled"]
        _di = "✅" if _rec["deviation"] < 0.3 else ("⚠️" if _rec["deviation"] < 0.6 else "❌")
        _tbl_ranked.append({
            "Rank":         _i + 1,
            "Fan":          _rec["fan_name"],
            "Motor":        _rec["motor_label"],
            "Angle (°)":    _rec["angle"],
            "Volume (CMH)": round(_sc["Q_CMH"]),
            "\u0394 CMH":        f"{_sc['Q_CMH'] - req_cmh:+.0f}",
            "FSP (mm WG)":  round(_sc["FSP"], 2),
            "\u0394 SP":         f"{_sc['FSP'] - req_sp:+.2f}",
            "Power (W)":    round(_sc["BKW"] * 1000, 1),
            "\u03b7 Static (%)": round(_sc["Static_Eff"], 1),
            "\u03b7 Total (%)":  round(_sc["Total_Eff"], 1),
            "Deviation":    f"{_di} {_rec['deviation']:.1%}",
            "Model":        _rec["model_name"],
            "CV R\u00b2":        f"{_rec['avg_r2_cv']:.4f}",
        })
    _tbl_ranked_df = pd.DataFrame(_tbl_ranked)
    st.dataframe(_tbl_ranked_df, use_container_width=True, hide_index=True)
    st.download_button(
        "\U0001f4e5 Download Results CSV",
        _tbl_ranked_df.to_csv(index=False),
        "cross_fan_selection.csv",
        "text/csv",
        key="cfs_download_ranked",
    )

    # ── Full predicted curves — top reasonable models only ────────────────────
    from plots import create_ml_prediction_curves
    from model_store import predict_for_fan

    top_recs = [r for r in recommendations if r["deviation"] < 0.6]
    if top_recs:
        st.markdown("---")
        st.markdown("### \U0001f4c8 Full Predicted Curves — Top Matching Models")
        for rec in top_recs:
            fid      = rec["fan_id"]
            computed = computed_map.get(fid)
            _pfx     = "\U0001f3c6 " if rec["recommended"] else ""
            label    = f"{_pfx}{rec['fan_name']} \u2014 {rec['motor_label']} \u2014 {rec['angle']}\u00b0"
            with st.expander(label, expanded=rec["recommended"]):
                try:
                    pred_df = predict_for_fan(fid, computed, rec["angle"])
                    fig     = create_ml_prediction_curves(pred_df, computed, rec["angle"])
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"cfs_curve_{fid}_{rec['angle']}_{rec['motor_label']}")
                except Exception as e:
                    st.warning(f"Could not render curve: {e}")
