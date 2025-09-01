# bond_model_dash.py
"""
Bond Model Dashboard (Dash)
---------------------------
A simple, educational dashboard that lets you explore how macro inputs and
supply/demand shocks can shift a sovereign yield curve and impact a specific bond's
price, yield, duration, and convexity.

How to run (Windows):
1) (Optional) Create a virtual environment:
   py -m venv venv
   venv\Scripts\activate

2) Install dependencies:
   pip install -r requirements.txt
   # or: pip install dash pandas plotly numpy

3) Run the app:
   python bond_model_dash.py

4) Open your browser at:
   http://127.0.0.1:8050

Notes:
- This is a stylised educational model, not investment advice.
- Elasticities are toy parameters to illustrate intuition. Adjust to taste.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go

# ----------------------
# Core bond math helpers
# ----------------------

def price_from_ytm(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    """
    Price of a plain-vanilla fixed coupon bond (clean-ish, ignoring accrued).
    coupon_rate_pct, ytm_pct are in percent (e.g., 4.5 means 4.5%).
    """
    c = coupon_rate_pct / 100.0
    y = ytm_pct / 100.0
    n = int(round(years * freq))
    if n <= 0:
        return face  # Immediate maturity edge case

    cf_coupon = face * c / freq
    df = [(1 + y / freq) ** t for t in range(1, n + 1)]
    pv_coupons = sum(cf_coupon / d for d in df)
    pv_redemption = face / df[-1]
    return pv_coupons + pv_redemption


def ytm_from_price(face: float, coupon_rate_pct: float, price: float, years: float, freq: int = 2, guess_pct: float = 4.0) -> float:
    """
    Solve for YTM (percent) given price using Newton-Raphson.
    Returns percent.
    """
    # Guard
    if years <= 0:
        return 0.0

    c = coupon_rate_pct / 100.0
    n = int(round(years * freq))
    cf_coupon = face * c / freq

    # If price ~ face and coupons ~ 0, return 0
    if abs(price - face) < 1e-6 and cf_coupon < 1e-9:
        return 0.0

    y = max(1e-9, guess_pct / 100.0)
    for _ in range(100):
        denom = [(1 + y / freq) ** t for t in range(1, n + 1)]
        f = sum(cf_coupon / d for d in denom) + face / denom[-1] - price

        # Derivative dPrice/dy (with respect to y, not percentage)
        dfdy = sum(- (t / freq) * cf_coupon / ((1 + y / freq) ** (t + 1)) for t in range(1, n + 1)) \
             + - (n / freq) * face / ((1 + y / freq) ** (n + 1))
        if abs(dfdy) < 1e-12:
            break
        y_new = y - f / dfdy
        if abs(y_new - y) < 1e-10:
            y = y_new
            break
        y = max(1e-9, y_new)

    return y * 100.0


def macaulay_duration(face: float, coupon_rate_pct: float, ytm_pct: float, years: float, freq: int = 2) -> float:
    """
    Macaulay duration in years.
    """
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
    """
    Approximate convexity in years^2.
    """
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


# ----------------------
# Yield curve construction
# ----------------------

def term_premium_component(years: np.ndarray, tp_max_pct: float = 1.0, beta: float = 3.0) -> np.ndarray:
    """
    Simple saturating term premium in percent.
    tp_max_pct = maximum premium (in %) as maturity goes to infinity.
    beta controls speed of saturation.
    """
    years = np.maximum(0.0, years)
    return tp_max_pct * (1.0 - np.exp(-years / beta))


def curve_spread_bps(
    issuance_bn: float,
    qe_bn: float,
    demand_idx: float,
    inflation_exp_pct: float,
    baseline_infl_target_pct: float = 2.0,
    supply_elasticity_bps_per_100bn: float = 5.0,  # +5 bps per +100bn net issuance
    demand_elasticity_bps_per_index: float = -10.0,  # +1 demand index -> -10 bps
    inflation_elasticity_bps_per_pct: float = 25.0   # +1% over target -> +25 bps
) -> float:
    """
    Returns a LEVEL shift (in basis points) applied to the whole curve.
    """
    net_supply_100bn = (issuance_bn - qe_bn) / 100.0
    supply_component = supply_elasticity_bps_per_100bn * net_supply_100bn
    demand_component = demand_elasticity_bps_per_index * demand_idx
    inflation_component = inflation_elasticity_bps_per_pct * (inflation_exp_pct - baseline_infl_target_pct)
    return supply_component + demand_component + inflation_component


def build_yield_curve(
    base_rate_pct: float,
    inflation_exp_pct: float,
    issuance_bn: float,
    qe_bn: float,
    demand_idx: float,
    risk_spread_bps: float,
    tp_max_pct: float = 1.0,
    maturities: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Construct a stylised yield curve (in percent) across maturities.
    """
    if maturities is None:
        maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)

    # Baseline: policy rate at short end + term premium
    baseline = base_rate_pct + term_premium_component(maturities, tp_max_pct=tp_max_pct, beta=3.0)

    # Level shift from supply/demand/inflation
    level_shift_bps = curve_spread_bps(
        issuance_bn=issuance_bn,
        qe_bn=qe_bn,
        demand_idx=demand_idx,
        inflation_exp_pct=inflation_exp_pct,
    )
    level_shift_pct = level_shift_bps / 100.0

    # Add any risk spread
    curve = baseline + level_shift_pct + (risk_spread_bps / 100.0)

    df = pd.DataFrame({
        "Maturity (Y)": maturities,
        "Yield (%)": curve
    })
    return df


# ----------------------
# Dash app
# ----------------------

app = Dash(__name__)
app.title = "Bond Model Dashboard"

def make_controls():
    return html.Div(
        [
            html.H3("Macro & Market Inputs", className="mt-4"),
            html.Label("Base policy rate (%)"),
            dcc.Slider(id="base_rate", min=-1.0, max=8.0, step=0.05, value=4.50,
                       tooltip={"placement": "bottom", "always_visible": False}),

            html.Label("Inflation expectations (%)"),
            dcc.Slider(id="infl_exp", min=0.0, max=10.0, step=0.1, value=3.0),

            html.Label("Gross issuance (bn)"),
            dcc.Slider(id="issuance", min=0.0, max=600.0, step=10.0, value=200.0),

            html.Label("QE (negative supply) (bn)"),
            dcc.Slider(id="qe", min=0.0, max=600.0, step=10.0, value=0.0),

            html.Label("Demand shock (−5 weak  …  +5 strong)"),
            dcc.Slider(id="demand_idx", min=-5.0, max=5.0, step=0.5, value=0.0),

            html.Label("Risk spread (bps)"),
            dcc.Slider(id="risk_spread_bps", min=-100.0, max=400.0, step=5.0, value=0.0),

            html.Label("Max term premium at long end (%)"),
            dcc.Slider(id="tp_max", min=0.0, max=3.0, step=0.05, value=1.0),

            html.Hr(),
            html.H3("Bond Settings"),
            html.Label("Maturity (years)"),
            dcc.Slider(id="bond_mty", min=0.5, max=30.0, step=0.5, value=10.0),

            html.Label("Coupon rate (%)"),
            dcc.Slider(id="bond_coupon", min=0.0, max=12.0, step=0.1, value=4.0),

            html.Label("Face value"),
            dcc.Slider(id="bond_face", min=50.0, max=1000.0, step=10.0, value=100.0),

            html.Label("Coupon frequency (per year)"),
            dcc.Slider(id="bond_freq", min=1, max=4, step=1, value=2,
                       marks={1: "Annual", 2: "Semi", 4: "Quarterly"}),
        ],
        style={"display": "grid", "rowGap": "10px"}
    )


def make_layout():
    return html.Div(
        [
            html.H1("Bond Model Dashboard"),
            html.P("Explore how issuance, QE, demand, and inflation expectations move yields "
                   "and affect a bond's price, duration, and convexity. (Educational model)"),
            html.Div(
                [
                    html.Div(make_controls(), style={"flex": "1", "minWidth": "320px", "maxWidth": "520px", "paddingRight": "20px"}),
                    html.Div(
                        [
                            dcc.Graph(id="yield_curve_graph"),
                            dcc.Graph(id="price_vs_yield_graph"),
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
def update_dashboard(base_rate, infl_exp, issuance, qe, demand_idx, risk_spread_bps, tp_max, bond_mty, bond_coupon, bond_face, bond_freq):
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

    # Interpolate yield for the chosen bond maturity
    x = curve_df["Maturity (Y)"].values
    y = curve_df["Yield (%)"].values
    ytm_at_mty = float(np.interp(bond_mty, x, y))

    # Price and risk
    price = price_from_ytm(face=bond_face, coupon_rate_pct=bond_coupon, ytm_pct=ytm_at_mty, years=bond_mty, freq=int(bond_freq))
    mac_dur = macaulay_duration(face=bond_face, coupon_rate_pct=bond_coupon, ytm_pct=ytm_at_mty, years=bond_mty, freq=int(bond_freq))
    mod_dur = modified_duration(face=bond_face, coupon_rate_pct=bond_coupon, ytm_pct=ytm_at_mty, years=bond_mty, freq=int(bond_freq))
    conv = convexity(face=bond_face, coupon_rate_pct=bond_coupon, ytm_pct=ytm_at_mty, years=bond_mty, freq=int(bond_freq))

    # Yield curve figure
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Maturity (Y)"], y=curve_df["Yield (%)"],
                                   mode="lines+markers", name="Yield Curve"))
    fig_curve.update_layout(title="Stylised Yield Curve",
                            xaxis_title="Maturity (years)",
                            yaxis_title="Yield (%)",
                            hovermode="x unified")

    # Price vs Yield (for the chosen bond)
    ytm_range = np.linspace(max(-1.0, ytm_at_mty - 3.0), ytm_at_mty + 3.0, 200)
    prices = [price_from_ytm(bond_face, bond_coupon, y, bond_mty, int(bond_freq)) for y in ytm_range]

    fig_pvy = go.Figure()
    fig_pvy.add_trace(go.Scatter(x=ytm_range, y=prices, mode="lines", name="Price vs Yield"))
    fig_pvy.add_trace(go.Scatter(x=[ytm_at_mty], y=[price], mode="markers", name="Current",
                                 marker=dict(size=10)))
    fig_pvy.update_layout(title=f"Price vs Yield (Bond: {bond_mty:.1f}Y, {bond_coupon:.2f}% coupon)",
                          xaxis_title="Yield to Maturity (%)",
                          yaxis_title="Price",
                          hovermode="x")

    # Metrics block
    metrics = html.Div(
        [
            html.H3("Bond Metrics"),
            html.Ul(
                [
                    html.Li(f"Interpolated YTM at {bond_mty:.1f}y: {ytm_at_mty:.3f}%"),
                    html.Li(f"Price (face {bond_face:.0f}): {price:.3f}"),
                    html.Li(f"Macaulay Duration: {mac_dur:.3f} years"),
                    html.Li(f"Modified Duration: {mod_dur:.3f} years"),
                    html.Li(f"Convexity: {conv:.3f} years²"),
                ]
            ),
        ]
    )

    # Assumptions text
    level_shift_bps = curve_spread_bps(
        issuance_bn=issuance, qe_bn=qe, demand_idx=demand_idx, inflation_exp_pct=infl_exp
    )
    assumptions = html.Div(
        [
            html.H3("Assumptions (Stylised)"),
            html.Ul(
                [
                    html.Li(f"Short end anchored by base rate = {base_rate:.2f}%"),
                    html.Li(f"Term premium rises to ~{tp_max:.2f}% at long end (saturating)"),
                    html.Li(f"Level shift from supply/demand/inflation = {level_shift_bps:.1f} bps"),
                    html.Li(f"Additional risk spread applied = {risk_spread_bps:.1f} bps"),
                ]
            ),
            html.P("Interpretation guide: ↑ issuance or ↓ demand → higher yields; ↑ QE or ↑ demand → lower yields; "
                   "higher inflation expectations push yields up vs a 2% target."),
        ]
    )

    return fig_curve, fig_pvy, metrics, assumptions


if __name__ == "__main__":
    app.run_server(debug=True)
