"""
plots.py — Interactive Plotly visualisations for fan performance data.

Every function returns a plotly.graph_objects.Figure (or a tuple with
auxiliary data) that can be rendered directly with st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata, interp1d

# ────────────────────────────────────────────────────────────────
# Visual constants
# ────────────────────────────────────────────────────────────────
ANGLE_COLORS = {
    20: '#00D4FF',   # Cyan
    30: '#00FF85',   # Green
    35: '#FFD700',   # Gold
    40: '#FF6B35',   # Orange
    45: '#FF2D55',   # Red-Pink
}

_CHART_BG  = 'rgba(17, 17, 28, 0.8)'
_PAPER_BG  = 'rgba(17, 17, 28, 0.0)'
_GRID      = 'rgba(255, 255, 255, 0.08)'
_FONT_CLR  = '#E0E0E0'
_PRED_CLR  = '#FF6BFF'   # magenta for ML predictions


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def _base_layout(title, xtitle, ytitle, height=500):
    return dict(
        template='plotly_dark',
        title=dict(text=title, font=dict(size=18, color=_FONT_CLR)),
        xaxis=dict(title=xtitle, gridcolor=_GRID, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=_GRID, zeroline=False),
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(color=_FONT_CLR, family='Inter, sans-serif'),
        height=height,
        legend=dict(bgcolor='rgba(0,0,0,0.3)',
                    bordercolor='rgba(255,255,255,0.15)', borderwidth=1),
        hovermode='x unified',
        margin=dict(t=60, b=60, l=60, r=40),
    )


def get_angle_color(angle: float) -> str:
    if angle in ANGLE_COLORS:
        return ANGLE_COLORS[angle]
    keys = sorted(ANGLE_COLORS)
    if angle <= keys[0]:
        return ANGLE_COLORS[keys[0]]
    if angle >= keys[-1]:
        return ANGLE_COLORS[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= angle <= keys[i + 1]:
            t  = (angle - keys[i]) / (keys[i + 1] - keys[i])
            c1, c2 = ANGLE_COLORS[keys[i]], ANGLE_COLORS[keys[i + 1]]
            r = int(int(c1[1:3], 16) * (1 - t) + int(c2[1:3], 16) * t)
            g = int(int(c1[3:5], 16) * (1 - t) + int(c2[3:5], 16) * t)
            b = int(int(c1[5:7], 16) * (1 - t) + int(c2[5:7], 16) * t)
            return f'#{r:02x}{g:02x}{b:02x}'
    return '#FFFFFF'


def _angle_traces(df, x_col, y_col, fig, hover_fmt=None, row=None, col=None,
                  show_legend=True, dash=None):
    """Add one trace per angle to *fig*."""
    for angle in sorted(df['ANGLE'].unique()):
        d = df[df['ANGLE'] == angle].sort_values(x_col)
        c = get_angle_color(angle)
        kw = dict(
            x=d[x_col], y=d[y_col],
            mode='lines+markers', name=f'{angle}°',
            line=dict(color=c, width=2.5, dash=dash),
            marker=dict(color=c, size=7),
            legendgroup=str(angle), showlegend=show_legend,
        )
        if hover_fmt:
            kw['hovertemplate'] = hover_fmt
        if row is not None:
            fig.add_trace(go.Scatter(**kw), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**kw))


# ────────────────────────────────────────────────────────────────
# 1.  Fan Curve  (Q vs FSP)
# ────────────────────────────────────────────────────────────────
def create_fan_curve(df):
    fig = go.Figure()
    _angle_traces(df, 'Q_CMH', 'FSP', fig,
                  'Q: %{x:.0f} CMH<br>FSP: %{y:.2f} mm WG<extra></extra>')
    fig.update_layout(**_base_layout(
        '📊 Fan Curve — Volume vs Static Pressure',
        'Volume Flow Rate (CMH)', 'Fan Static Pressure (mm WG)'))
    return fig


# ────────────────────────────────────────────────────────────────
# 2.  FTP Curve
# ────────────────────────────────────────────────────────────────
def create_ftp_curve(df):
    fig = go.Figure()
    _angle_traces(df, 'Q_CMH', 'FTP', fig,
                  'Q: %{x:.0f} CMH<br>FTP: %{y:.2f} mm WG<extra></extra>')
    fig.update_layout(**_base_layout(
        '📊 Fan Total Pressure Curve',
        'Volume Flow Rate (CMH)', 'Fan Total Pressure (mm WG)'))
    return fig


# ────────────────────────────────────────────────────────────────
# 3.  Power Curve
# ────────────────────────────────────────────────────────────────
def create_power_curve(df):
    fig = go.Figure()
    for angle in sorted(df['ANGLE'].unique()):
        d = df[df['ANGLE'] == angle].sort_values('Q_CMH')
        c = get_angle_color(angle)
        fig.add_trace(go.Scatter(
            x=d['Q_CMH'], y=d['BKW'] * 1000,
            mode='lines+markers', name=f'{angle}°',
            line=dict(color=c, width=2.5),
            marker=dict(color=c, size=7),
            hovertemplate='Q: %{x:.0f} CMH<br>Power: %{y:.1f} W<extra></extra>',
        ))
    fig.update_layout(**_base_layout(
        '⚡ Power Curve — Volume vs Brake Power',
        'Volume Flow Rate (CMH)', 'Brake Power (W)'))
    return fig


# ────────────────────────────────────────────────────────────────
# 4.  Efficiency Curves  (static & total side by side)
# ────────────────────────────────────────────────────────────────
def create_efficiency_curves(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Static Efficiency', 'Total Efficiency'],
                        horizontal_spacing=0.1)
    _angle_traces(df, 'Q_CMH', 'Static_Eff', fig, row=1, col=1)
    _angle_traces(df, 'Q_CMH', 'Total_Eff',  fig, row=1, col=2, show_legend=False)

    lo = _base_layout('🎯 Efficiency Curves', '', '', height=480)
    lo.pop('xaxis', None); lo.pop('yaxis', None)
    fig.update_layout(**lo)
    for c in (1, 2):
        fig.update_xaxes(title_text='Volume (CMH)', gridcolor=_GRID, row=1, col=c)
    fig.update_yaxes(title_text='Static Eff (%)', gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Total Eff (%)',  gridcolor=_GRID, row=1, col=2)
    return fig


# ────────────────────────────────────────────────────────────────
# 5.  Combined Performance (single angle, dual Y)
# ────────────────────────────────────────────────────────────────
def create_combined_performance(df, angle):
    d = df[df['ANGLE'] == angle].sort_values('Q_CMH')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=d['Q_CMH'], y=d['FSP'], mode='lines+markers', name='FSP',
        line=dict(color='#00D4FF', width=3), marker=dict(size=8)),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=d['Q_CMH'], y=d['FTP'], mode='lines+markers', name='FTP',
        line=dict(color='#00FF85', width=3, dash='dash'), marker=dict(size=8)),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=d['Q_CMH'], y=d['Static_Eff'], mode='lines+markers',
        name='Static Eff', line=dict(color='#FFD700', width=2.5),
        marker=dict(size=7, symbol='diamond')),
        secondary_y=True)
    fig.add_trace(go.Scatter(
        x=d['Q_CMH'], y=d['Total_Eff'], mode='lines+markers',
        name='Total Eff', line=dict(color='#FF6B35', width=2.5, dash='dot'),
        marker=dict(size=7, symbol='diamond')),
        secondary_y=True)

    lo = _base_layout(f'📈 Combined Performance — {angle}° Blade Angle',
                       'Volume Flow Rate (CMH)', '', height=550)
    lo.pop('yaxis', None)
    fig.update_layout(**lo)
    fig.update_yaxes(title_text='Pressure (mm WG)', secondary_y=False, gridcolor=_GRID)
    fig.update_yaxes(title_text='Efficiency (%)',    secondary_y=True,  gridcolor=_GRID)
    return fig


# ────────────────────────────────────────────────────────────────
# 6.  Angle-wise Comparison (bar charts)
# ────────────────────────────────────────────────────────────────
def create_angle_comparison(df):
    rows = []
    for angle in sorted(df['ANGLE'].unique()):
        d = df[df['ANGLE'] == angle]
        rows.append({
            'Angle': angle,
            'Max Volume (CMH)':   d['Q_CMH'].max(),
            'Max FSP (mm WG)':    d['FSP'].max(),
            'Max Static Eff (%)': d['Static_Eff'].max(),
            'Max Total Eff (%)':  d['Total_Eff'].max(),
            'Max BKW (W)':        d['BKW'].max() * 1000,
            'BEP Volume (CMH)':   d.loc[d['Total_Eff'].idxmax(), 'Q_CMH'],
        })
    sdf = pd.DataFrame(rows)
    labels = [f"{a}°" for a in sdf['Angle']]
    colors = [get_angle_color(a) for a in sdf['Angle']]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Max Volume', 'Max FSP',
                        'Peak Efficiency', 'BEP Volume'],
        vertical_spacing=0.18, horizontal_spacing=0.12)

    def _bar(vals, r, c):
        fig.add_trace(go.Bar(
            x=labels, y=vals, marker_color=colors, showlegend=False,
            text=[f'{v:.0f}' for v in vals], textposition='outside'),
            row=r, col=c)

    _bar(sdf['Max Volume (CMH)'].values,   1, 1)
    _bar(sdf['Max FSP (mm WG)'].values,    1, 2)
    _bar(sdf['Max Total Eff (%)'].values,  2, 1)
    _bar(sdf['BEP Volume (CMH)'].values,   2, 2)

    lo = _base_layout('📊 Angle-wise Performance Comparison', '', '', height=620)
    lo.pop('xaxis', None); lo.pop('yaxis', None)
    fig.update_layout(**lo)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(gridcolor=_GRID, row=r, col=c)
            fig.update_yaxes(gridcolor=_GRID, row=r, col=c)
    return fig, sdf


# ────────────────────────────────────────────────────────────────
# 7.  3-D Surface  (Angle × Volume → target)
# ────────────────────────────────────────────────────────────────
def create_3d_surface(df, target='FSP', title=None):
    title = title or f'3D Surface — {target}'
    n_angles = df['ANGLE'].nunique()
    # For very few angles, reduce resolution to avoid NaN-dominated grids
    grid_n = 50 if n_angles >= 3 else 30
    a_rng = np.linspace(df['ANGLE'].min(), df['ANGLE'].max(), grid_n)
    v_rng = np.linspace(df['Q_CMH'].min(), df['Q_CMH'].max(), grid_n)
    ag, vg = np.meshgrid(a_rng, v_rng)
    pts = df[['ANGLE', 'Q_CMH']].values
    zvals = df[target].values
    # Try cubic first, fall back to linear then nearest for sparse data
    zg = None
    for method in ('cubic', 'linear', 'nearest'):
        zg = griddata(pts, zvals, (ag, vg), method=method)
        if not np.all(np.isnan(zg)):
            break

    fig = go.Figure(go.Surface(
        x=ag, y=vg, z=zg, colorscale='Viridis', opacity=0.92,
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor='limegreen', project_z=True))))
    fig.update_layout(
        title=dict(text=f'🌐 {title}', font=dict(size=18, color=_FONT_CLR)),
        scene=dict(xaxis_title='Blade Angle (°)',
                   yaxis_title='Volume (CMH)', zaxis_title=target,
                   bgcolor=_CHART_BG,
                   xaxis=dict(gridcolor=_GRID),
                   yaxis=dict(gridcolor=_GRID),
                   zaxis=dict(gridcolor=_GRID)),
        template='plotly_dark', paper_bgcolor=_PAPER_BG,
        font=dict(color=_FONT_CLR, family='Inter, sans-serif'),
        height=620, margin=dict(t=50, b=30, l=30, r=30))
    return fig


# ────────────────────────────────────────────────────────────────
# 8.  Predicted vs Actual scatter  (per model, LOOCV)
# ────────────────────────────────────────────────────────────────
def create_prediction_vs_actual(df, model_info, target='Q_CMH'):
    fig = go.Figure()
    idx = model_info['target_cols'].index(target)

    for name, res in model_info['results'].items():
        actual  = df[target].values
        pred    = res['loo_predictions'][:, idx]
        r2      = res['r2_cv'][target]
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode='markers',
            name=f'{name}  (R²={r2:.3f})',
            marker=dict(size=10, opacity=0.85)))

    mn, mx = df[target].min(), df[target].max()
    pad = (mx - mn) * 0.1
    fig.add_trace(go.Scatter(
        x=[mn - pad, mx + pad], y=[mn - pad, mx + pad],
        mode='lines', name='Ideal',
        line=dict(color='rgba(255,255,255,0.25)', dash='dash', width=1)))
    fig.update_layout(**_base_layout(
        f'🎯 Predicted vs Actual — {target}',
        f'Actual {target}', f'Predicted {target}'))
    return fig


# ────────────────────────────────────────────────────────────────
# 9.  ML prediction curves at a custom angle (2×2 subplots)
# ────────────────────────────────────────────────────────────────
def create_ml_prediction_curves(pred_df, actual_df=None, angle=None):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Volume vs Static Pressure',
                        'Volume vs Power',
                        'Volume vs Static Efficiency',
                        'Volume vs Total Efficiency'],
        vertical_spacing=0.14, horizontal_spacing=0.12)

    ps = pred_df.sort_values('Q_CMH')

    def _add_pred(y_col, r, c, y_mul=1, sl=True):
        fig.add_trace(go.Scatter(
            x=ps['Q_CMH'], y=ps[y_col] * y_mul,
            mode='lines', name=f'Predicted ({angle}°)',
            line=dict(color=_PRED_CLR, width=3),
            legendgroup='pred', showlegend=sl), row=r, col=c)

    _add_pred('FSP',        1, 1)
    _add_pred('BKW',        1, 2, 1000, False)
    _add_pred('Static_Eff', 2, 1, sl=False)
    _add_pred('Total_Eff',  2, 2, sl=False)

    if actual_df is not None:
        for a in sorted(actual_df['ANGLE'].unique()):
            d = actual_df[actual_df['ANGLE'] == a].sort_values('Q_CMH')
            c = get_angle_color(a)
            kw = dict(mode='markers+lines',
                      line=dict(color=c, width=1.5, dash='dot'),
                      marker=dict(color=c, size=5), opacity=0.55,
                      legendgroup=f'a{a}')
            fig.add_trace(go.Scatter(x=d['Q_CMH'], y=d['FSP'],
                          name=f'Actual {a}°', **kw), row=1, col=1)
            fig.add_trace(go.Scatter(x=d['Q_CMH'], y=d['BKW']*1000,
                          showlegend=False, **kw), row=1, col=2)
            fig.add_trace(go.Scatter(x=d['Q_CMH'], y=d['Static_Eff'],
                          showlegend=False, **kw), row=2, col=1)
            fig.add_trace(go.Scatter(x=d['Q_CMH'], y=d['Total_Eff'],
                          showlegend=False, **kw), row=2, col=2)

    lo = _base_layout(f'🤖 ML Predicted Performance — {angle}° Blade Angle',
                       '', '', height=700)
    lo.pop('xaxis', None); lo.pop('yaxis', None)
    fig.update_layout(**lo)
    for c in (1, 2):
        fig.update_xaxes(title_text='Volume (CMH)', gridcolor=_GRID, row=1, col=c)
        fig.update_xaxes(title_text='Volume (CMH)', gridcolor=_GRID, row=2, col=c)
    fig.update_yaxes(title_text='FSP (mm WG)',   gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Power (W)',     gridcolor=_GRID, row=1, col=2)
    fig.update_yaxes(title_text='Static Eff (%)', gridcolor=_GRID, row=2, col=1)
    fig.update_yaxes(title_text='Total Eff (%)',  gridcolor=_GRID, row=2, col=2)
    return fig


# ────────────────────────────────────────────────────────────────
# 10.  System Resistance Overlay
# ────────────────────────────────────────────────────────────────
def create_system_resistance_overlay(df, angle, k_sys):
    d = df[df['ANGLE'] == angle].sort_values('Q_CMH')
    c = get_angle_color(angle)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d['Q_CMH'], y=d['FSP'], mode='lines+markers',
        name=f'Fan Curve ({angle}°)',
        line=dict(color=c, width=3), marker=dict(size=8)))

    q_rng  = np.linspace(0, d['Q_CMH'].max() * 1.2, 200)
    sp_sys = k_sys * q_rng**2
    fig.add_trace(go.Scatter(
        x=q_rng, y=sp_sys, mode='lines',
        name=f'System (k={k_sys:.2e})',
        line=dict(color=_PRED_CLR, width=2.5, dash='dash')))

    # operating-point intersection
    if len(d) >= 2:
        try:
            f_fan = interp1d(d['Q_CMH'].values, d['FSP'].values,
                             kind='linear', fill_value='extrapolate')
            qf = np.linspace(d['Q_CMH'].min(), d['Q_CMH'].max(), 1000)
            diff = np.abs(f_fan(qf) - k_sys * qf**2)
            oi = np.argmin(diff)
            fig.add_trace(go.Scatter(
                x=[qf[oi]], y=[f_fan(qf[oi])], mode='markers',
                name=f'OP  ({qf[oi]:.0f} CMH, {f_fan(qf[oi]):.1f} mm WG)',
                marker=dict(color='#FFF', size=14, symbol='star',
                            line=dict(color='#FFD700', width=2))))
        except Exception:
            pass

    fig.update_layout(**_base_layout(
        f'🔄 System Resistance Overlay — {angle}°',
        'Volume Flow Rate (CMH)', 'Pressure (mm WG)'))
    return fig
