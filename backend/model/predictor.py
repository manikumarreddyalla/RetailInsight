import numpy as np
import pandas as pd
from datetime import timedelta

from utils import prepare_dataset, load_raw_data

# Load merged dataset once
df = prepare_dataset()
# Load raw to get calendar if needed later
_, _, calendar_df = load_raw_data()


def _compute_trend(values, max_window=90):
    """Simple linear trend using last N points."""
    n = len(values)
    if n < 2:
        return 0.0, float(values[-1])

    window = min(n, max_window)
    y = np.array(values[-window:], dtype=float)
    x = np.arange(window, dtype=float)

    # Fit y = a*x + b
    a, b = np.polyfit(x, y, 1)
    return a, b + a * (window - 1)  # slope, last_trend_level


def _compute_seasonality_by_dow(df_p):
    """Average quantity per day-of-week."""
    tmp = df_p.copy()
    tmp["dow"] = tmp["date"].dt.dayofweek
    return tmp.groupby("dow")["quantity_sold"].mean().to_dict()


def forecast_product(product_id: str, horizon: int = 30):
    """
    Lightweight MATPFN-style forecast:
    - Uses last 30 days as level signal
    - Uses linear trend from last up to 90 days
    - Uses day-of-week seasonality
    - Fuses them with fixed weights (no heavy model)
    """

    df_p = df[df["product_id"] == product_id].copy().sort_values("date")

    if df_p.empty:
        return {"error": "No sales data for this product."}

    if len(df_p) < 30:
        return {"error": "Not enough history (<30 days) for this product."}

    # === BASE SERIES ===
    qty = df_p["quantity_sold"].astype(float).tolist()
    last_30 = qty[-30:]
    last_date = df_p["date"].max()

    overall_mean = float(np.mean(qty))
    dow_means = _compute_seasonality_by_dow(df_p)

    # Trend from last 90 points max
    slope, last_trend_level = _compute_trend(qty, max_window=90)

    # Start moving window for level component
    window_values = last_30.copy()

    future_values = []
    future_dates = []

    # Weights for fusion (can tune)
    w_level = 0.4
    w_trend = 0.4
    w_season = 0.2

    n_past = len(qty)

    for step in range(1, horizon + 1):
        future_date = last_date + timedelta(days=step)
        dow = future_date.weekday()

        # Level: mean of last 7 (or all if less)
        if len(window_values) >= 7:
            level_component = float(np.mean(window_values[-7:]))
        else:
            level_component = float(np.mean(window_values))

        # Trend: extrapolate using slope from past
        trend_index = n_past + step - 1  # continue index
        trend_component = last_trend_level + slope * (step)

        # Seasonality: dow adjustment
        dow_mean = dow_means.get(dow, overall_mean)
        season_component = dow_mean

        # MATPFN-style fusion
        y_hat = (
            w_level * level_component
            + w_trend * trend_component
            + w_season * season_component
        )

        # Safety: no negative demand
        y_hat = max(0.0, float(y_hat))

        future_values.append(y_hat)
        future_dates.append(future_date)

        # Update window with predicted value
        window_values.append(y_hat)
        if len(window_values) > 30:
            window_values.pop(0)

    # HISTORY for last 120 days (for debugging / future UI)
    history_df = (
        df_p.tail(120)[["date", "quantity_sold"]]
        .rename(columns={"quantity_sold": "quantity"})
    )

    return {
        "product_id": product_id,
        # For your current UI we just need numeric list:
        "forecast": future_values,
        # History can be used later if you want:
        "history": history_df.to_dict(orient="records"),
    }
