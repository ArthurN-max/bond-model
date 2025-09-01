# bond_model_dash.py
# Dash 3.x + Plotly 6.x
# Real yield curve only (no demo trace), fixed chart sizes, neat slider marks, robust callback.

from __future__ import annotations
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# Create Dash app and expose WSGI server for Gunicorn
app = Dash(__name__)
server = app.server  # <— Render/Gunicorn entrypoint

# -----------------------------
# Bond maths (plain-vanilla)
# -----------------------------
def price_from_ytm(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    c = coupon_rate_pct / 100.0
    y = ytm_pct / 100.0
    n = int(round(years * freq))
    if n <= 0:
        return face
    cf_coupon = face * c / freq
    df = [(1 + y / freq) ** t for t in range(1, n + 1)]
    pv_coupons = sum(cf_coupon / d for d in df)
    pv_redemption = face / df[-1]
    return pv_coupons + pv_redemption

def macaulay_duration(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    c = coupon_rate_pct / 100.0
    y = ytm_pct / 100.0
    n = int(round(years * freq))
    if n <= 0:
        return 0.0
    cf_coupon = face * c / freq
    cf_times = np.arange(1, n + 1)
    df = (1 + y / freq) ** cf_times
    cashflows = np.full(n, cf_coupon, dtype=float)
    cashflows[-1] += face
    pv = np.sum(cashflows / df)
    if pv <= 0:
        return 0.0
    weighted_times = np.sum((cf_times / freq) * (cashflows / df))
    return weighted_times / pv

def modified_duration(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    mac = macaulay_duration(face, coupon_rate_pct, ytm_pct, years, freq)
    y = ytm_pct / 100.0
    return mac / (1 + y / freq)

def convexity(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    c = coupon_rate_pct / 100.0
    y = ytm_pct / 100.0
    n = int(round(years * freq))
    if n <= 0:
        return 0.0
    cf_coupon = face * c / freq
    cf_times = np.arange(1, n + 1)
    df = (1 + y / freq) ** cf_times
    cashflows = np.full(n, cf_coupon, dtype=float)
    cashflows[-1] += face
    terms = (cf_times * (cf_times + 1)) * (cashflows / (df * (1 + y / freq) ** 2))
    conv = np.sum(terms) / (freq ** 2)
    price = np.sum(cashflows / df)
    if price <= 0:
        return 0.0
    return conv / price

# -----------------------------
# Yield curve construction
# -----------------------------
def term_premium_component(years: np.ndarray, tp_max_pct: float = 1.0, beta: float = 3.0) -> np.ndarray:
    years = np.maximum(0.0, years)
    return tp_max_pct * (1.0 - np.exp(-years / beta))

def curve_spread_bps(
    issuance_bn: float, qe_bn: float, demand_idx: float, inflation_exp_pct: float,
    baseline_infl_target_pct: float = 2.0,
    supply_elasticity_bps_per_100bn: float = 5.0,
    demand_elasticity_bps_per_index: float = -10.0,
    inflation_elasticity_bps_per_pct: float = 25.0,
) -> float:
    net_supply_100bn = (issuance_bn - qe_bn) / 100.0
    supply_component = supply_elasticity_bps_per_100bn * net_supply_100bn
    demand_component = demand_elasticity_bps_per_index * demand_idx
    inflation_component = inflation_elasticity_bps_per_pct * (inflation_exp_pct - baseline_infl_target_pct)
    return supply_component + demand_component + inflation_component

def build_yield_curve(
    base_rate_pct: float, inflation_exp_pct: float, issuance_bn: float, qe_bn: float,
    demand_idx: float, risk_spread_bps: float, tp_max_pct: float = 1.0,
    maturities: np.ndarray | None = None,
) -> pd.DataFrame:
    if maturities is None:
        maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    baseline = base_rate_pct + term_premium_component(maturities, tp_max_pct=tp_max_pct, beta=3.0)
    level_shift_pct = curve_spread_bps(issuance_bn=issuance_bn, qe_bn=qe_bn, demand_idx=demand_idx,
                                       inflation_exp_pct=inflation_exp_pct) / 100.0
    curve = baseline + level_shift_pct + (risk_spread_bps / 100.0)
    return pd.DataFrame({"Maturity (Y)": maturities, "Yield (%)": curve})

# -----------------------------
# Dash app + layout
# -----------------------------
app.title = "Bond Model Dashboard"

def make_controls():
    return html.Div(
        [
            html.H3("Macro & Market Inputs"),
            html.Label("Base policy rate (%)"),
            dcc.Slider(id="base_rate", min=-1.0, max=8.0, step=0.05, value=4.50,
                       marks={-1: "-1%", 0: "0%", 2: "2%", 4: "4%", 6: "6%", 8: "8%"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Inflation expectations (%)"),
            dcc.Slider(id="infl_exp", min=0.0, max=10.0, step=0.1, value=3.0,
                       marks={0: "0%", 2: "2%", 4: "4%", 6: "6%", 8: "8%", 10: "10%"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Gross issuance (bn)"),
            dcc.Slider(id="issuance", min=0.0, max=600.0, step=10.0, value=200.0,
                       marks={0: "0", 100: "100", 200: "200", 300: "300", 400: "400", 500: "500", 600: "600"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("QE (negative supply) (bn)"),
            dcc.Slider(id="qe", min=0.0, max=600.0, step=10.0, value=0.0,
                       marks={0: "0", 100: "100", 200: "200", 300: "300", 400: "400", 500: "500", 600: "600"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Demand shock (−5 weak … +5 strong)"),
            dcc.Slider(id="demand_idx", min=-5.0, max=5.0, step=0.5, value=0.0,
                       marks={-5: "-5", -3: "-3", -1: "-1", 0: "0", 1: "1", 3: "3", 5: "5"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Risk spread (bps)"),
            dcc.Slider(id="risk_spread_bps", min=-100.0, max=400.0, step=5.0, value=0.0,
                       marks={-100: "-100", 0: "0", 100: "100", 200: "200", 300: "300", 400: "400"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Max term premium at long end (%)"),
            dcc.Slider(id="tp_max", min=0.0, max=3.0, step=0.05, value=1.0,
                       marks={0: "0%", 0.5: "0.5%", 1.0: "1%", 1.5: "1.5%", 2.0: "2%", 2.5: "2.5%", 3.0: "3%"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Hr(),
            html.H3("Bond Settings"),
            html.Label("Maturity (years)"),
            dcc.Slider(id="bond_mty", min=0.5, max=30.0, step=0.5, value=10.0,
                       marks={1: "1", 2: "2", 5: "5", 10: "10", 15: "15", 20: "20", 30: "30"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Coupon rate (%)"),
            dcc.Slider(id="bond_coupon", min=0.0, max=12.0, step=0.1, value=4.0,
                       marks={0: "0%", 2: "2%", 4: "4%", 6: "6%", 8: "8%", 10: "10%", 12: "12%"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Face value"),
            dcc.Slider(id="bond_face", min=50.0, max=1000.0, step=10.0, value=100.0,
                       marks={50: "50", 100: "100", 250: "250", 500: "500", 750: "750", 1000: "1000"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Coupon frequency (per year)"),
            dcc.Slider(id="bond_freq", min=1, max=4, step=1, value=2,
                       marks={1: "Annual", 2: "Semi", 4: "Quarterly"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ],
        style={"display": "grid", "rowGap": "10px"},
    )

def make_layout():
    return html.Div(
        [
            html.H1("Bond Model Dashboard"),
            html.P("Explore how issuance, QE, demand, and inflation expectations move yields "
                   "and affect a bond's price, duration, and convexity. (Educational model)"),
            html.Div(
                [
                    html.Div(make_controls(),
                             style={"flex": "1", "minWidth": "320px", "maxWidth": "520px", "paddingRight": "20px"}),
                    html.Div(
                        [
                            dcc.Graph(id="yield_curve_graph", style={"height": "480px", "width": "100%"}),
                            dcc.Graph(id="price_vs_yield_graph", style={"height": "420px", "width": "100%"}),
                            html.Div(id="bond_metrics"),
                            html.Hr(),
                            html.Div(id="assumptions_text"),
                        ],
                        style={"flex": "2", "minWidth": "420px"}
                    ),
                ],
                style={"display": "flex", "flexDirection": "row", "gap": "20px"}
            ),
            html.Footer("Not investment advice. v1.0", style={"marginTop": "20px", "opacity": 0.6}),
        ],
        style={"padding": "20px", "fontFamily": "Arial, sans-serif"}
    )

app.layout = make_layout()

# -----------------------------
# Robust callback (real curve only)
# -----------------------------
@app.callback(
    Output("yield_curve_graph", "figure"),
    Output("price_vs_yield_graph", "figure"),
    Output("bond_metrics", "children"),
    Output("assumptions_text", "children"),
    Input("base_rate", "value"),
    Input("infl_exp", "value"),
    Input("issuance", "value"),
    Input("qe", "value"),
    Input("demand_idx", "value"),
    Input("risk_spread_bps", "value"),
    Input("tp_max", "value"),
    Input("bond_mty", "value"),
    Input("bond_coupon", "value"),
    Input("bond_face", "value"),
    Input("bond_freq", "value"),
)
def update_dashboard(base_rate, infl_exp, issuance, qe, demand_idx, risk_spread_bps, tp_max,
                     bond_mty, bond_coupon, bond_face, bond_freq):

    # Safe casts
    base_rate        = float(base_rate or 0.0)
    infl_exp         = float(infl_exp or 0.0)
    issuance         = float(issuance or 0.0)
    qe               = float(qe or 0.0)
    demand_idx       = float(demand_idx or 0.0)
    risk_spread_bps  = float(risk_spread_bps or 0.0)
    tp_max           = float(tp_max or 0.0)
    bond_mty         = float(bond_mty or 10.0)
    bond_coupon      = float(bond_coupon or 0.0)
    bond_face        = float(bond_face or 100.0)
    bond_freq        = int(bond_freq or 2)

    # Build curve
    curve_df = build_yield_curve(
        base_rate_pct=base_rate,
        inflation_exp_pct=infl_exp,
        issuance_bn=issuance,
        qe_bn=qe,
        demand_idx=demand_idx,
        risk_spread_bps=risk_spread_bps,
        tp_max_pct=tp_max,
    )

    # Prepare x/y; ensure finite; fallback if needed
    x = np.asarray(curve_df["Maturity (Y)"].values, dtype=float)
    y = np.asarray(curve_df["Yield (%)"].values, dtype=float)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        x = np.array([1.0, 5.0, 10.0, 30.0], dtype=float)
        y = np.array([base_rate, base_rate + 0.5, base_rate + 1.0, base_rate + 1.2], dtype=float)

    # Interpolated YTM
    ytm_at_mty = float(np.interp(bond_mty, x, y))

    # Price & risk metrics
    price   = price_from_ytm(bond_face, bond_coupon, ytm_at_mty, bond_mty, bond_freq)
    mac_dur = macaulay_duration(bond_face, bond_coupon, ytm_at_mty, bond_mty, bond_freq)
    mod_dur = modified_duration(bond_face, bond_coupon, ytm_at_mty, bond_mty, bond_freq)
    conv    = convexity(bond_face, bond_coupon, ytm_at_mty, bond_mty, bond_freq)

    # --- Yield curve figure (real curve only) ---
    real_x = list(map(float, x))
    real_y = list(map(float, y))

    fig_curve = go.Figure(data=[
        go.Scatter(x=real_x, y=real_y, mode="lines+markers", name="Yield Curve", line=dict(width=3)),
    ])
    fig_curve.update_layout(
        title="Stylised Yield Curve",
        xaxis_title="Maturity (years)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=40),
        height=480,
    )
    y_min = float(np.nanmin(real_y))
    y_max = float(np.nanmax(real_y))
    ypad = max(0.2, 0.1 * max(1.0, y_max - y_min))
    fig_curve.update_yaxes(range=[y_min - ypad, y_max + ypad])

    # --- Price vs Yield ---
    y_left  = max(-1.0, ytm_at_mty - 3.0)
    y_right = ytm_at_mty + 3.0
    if y_right <= y_left:
        y_right = y_left + 1.0
    ytm_range = np.linspace(y_left, y_right, 200)
    prices = [price_from_ytm(bond_face, bond_coupon, yy, bond_mty, bond_freq) for yy in ytm_range]

    fig_pvy = go.Figure(data=[
        go.Scatter(x=list(ytm_range), y=prices, mode="lines", name="Price vs Yield", line=dict(width=3)),
        go.Scatter(x=[ytm_at_mty], y=[price], mode="markers", name="Current", marker=dict(size=10))
    ])
    fig_pvy.update_layout(
        title=f"Price vs Yield (Bond: {bond_mty:.1f}Y, {bond_coupon:.2f}% coupon)",
        xaxis_title="Yield to Maturity (%)",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x",
        margin=dict(l=50, r=20, t=50, b=40),
        height=420,
    )

    # Metrics & assumptions
    metrics = html.Div([
        html.H3("Bond Metrics"),
        html.Ul([
            html.Li(f"Interpolated YTM at {bond_mty:.1f}y: {ytm_at_mty:.3f}%"),
            html.Li(f"Price (face {bond_face:.0f}): {price:.3f}"),
            html.Li(f"Macaulay Duration: {mac_dur:.3f} years"),
            html.Li(f"Modified Duration: {mod_dur:.3f} years"),
            html.Li(f"Convexity: {conv:.3f} years²"),
        ]),
    ])

    level_shift_bps = curve_spread_bps(issuance_bn=issuance, qe_bn=qe, demand_idx=demand_idx, inflation_exp_pct=infl_exp)
    assumptions = html.Div([
        html.H3("Assumptions (Stylised)"),
        html.Ul([
            html.Li(f"Short end anchored by base rate = {base_rate:.2f}%"),
            html.Li(f"Term premium rises to ~{tp_max:.2f}% at long end (saturating)"),
            html.Li(f"Level shift from supply/demand/inflation = {level_shift_bps:.1f} bps"),
            html.Li(f"Additional risk spread applied = {risk_spread_bps:.1f} bps"),
        ]),
        html.P("Interpretation: ↑ issuance or ↓ demand → higher yields; ↑ QE or ↑ demand → lower yields; "
               "higher inflation expectations push yields up vs a 2% target."),
    ])

    return fig_curve, fig_pvy, metrics, assumptions

# Local dev runner (ignored by Gunicorn on Render)
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
