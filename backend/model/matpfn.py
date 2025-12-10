import numpy as np
import pandas as pd


class MATPFNModel:
    """
    Light-weight MATPFN style forecaster.

    Agents (all computed from the passed product dataframe):
      - Trend agent: linear trend over time
      - Seasonality agent: day-of-week pattern
      - Level agent: last 7-day average
      - Volatility agent: std used only to clip weird values

    We don't store a trained model on disk – everything is
    computed on-the-fly from the product's history (pdf).
    """

    @staticmethod
    def simple_forecast(pdf: pd.DataFrame, horizon: int = 30):
        """
        pdf: DataFrame with columns ['date', 'quantity_sold'] for ONE product.
        horizon: number of days to forecast.
        Returns: list[int] of length = horizon.
        """

        df = pdf.copy()

        # Basic safety checks
        if df.empty or "date" not in df.columns or "quantity_sold" not in df.columns:
            return []

        # Ensure types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        if len(df) < 5:
            # Too little data → repeat last value
            last_val = int(df["quantity_sold"].iloc[-1])
            return [last_val] * horizon

        # ======= TREND AGENT (linear regression on time) =======
        # numeric time index (days from first date)
        df["t"] = (df["date"] - df["date"].min()).dt.days
        y = df["quantity_sold"].values.astype(float)
        t = df["t"].values.astype(float)

        # If all t are same (rare), avoid polyfit crash
        if np.all(t == t[0]):
            trend_slope, trend_intercept = 0.0, float(y.mean())
        else:
            trend_slope, trend_intercept = np.polyfit(t, y, 1)

        # ======= SEASONALITY AGENT (day-of-week pattern) =======
        df["dow"] = df["date"].dt.dayofweek  # 0=Mon
        dow_means = df.groupby("dow")["quantity_sold"].mean()
        global_mean = df["quantity_sold"].mean()

        # ======= LEVEL / VOLATILITY AGENT =======
        window = min(7, len(df))
        level_mean = df["quantity_sold"].tail(window).mean()
        last_val = df["quantity_sold"].iloc[-1]
        vol = df["quantity_sold"].std() if len(df) > 1 else 0.0

        # ======= FUSION: generate future dates + combine agents =======
        last_date = df["date"].iloc[-1]
        min_date = df["date"].min()

        forecast_vals = []

        for i in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=i)
            t_future = (future_date - min_date).days

            # Trend prediction
            trend_pred = trend_slope * t_future + trend_intercept

            # Seasonality adjustment
            dow = future_date.dayofweek
            dow_mean = dow_means.get(dow, global_mean)
            seasonal_adj = dow_mean - global_mean  # deviation from overall

            # Level baseline
            level_pred = level_mean

            # ===== FUSION WEIGHTS (light MATPFN) =====
            # You can tune these 0.5 / 0.3 / 0.2 later.
            fused = (
                0.5 * trend_pred +
                0.3 * (level_pred + seasonal_adj) +
                0.2 * last_val
            )

            # Simple clipping to avoid absurd negatives
            if vol > 0:
                lower = global_mean - 3 * vol
                upper = global_mean + 3 * vol
                fused = max(lower, min(upper, fused))

            forecast_vals.append(max(0, int(round(fused))))

        return forecast_vals
