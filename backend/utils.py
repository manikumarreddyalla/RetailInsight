import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# ============================
# LOAD RAW DATA
# ============================
def load_raw_data():
    sales = pd.read_csv(os.path.join(DATA_PATH, "sales_dataset.csv"))
    products = pd.read_csv(os.path.join(DATA_PATH, "products_master.csv"))
    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar_dataset.csv"))

    sales["date"] = pd.to_datetime(sales["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])

    return sales, products, calendar


# ============================
# MERGE + FEATURE ENGINEERING
# ============================
def prepare_dataset():
    sales, products, calendar = load_raw_data()

    df = sales.merge(calendar, on="date", how="left")
    df = df.merge(products, on="product_id", how="left")

    df = df.sort_values(["product_id", "date"])

    # ---- LAGS ----
    df["lag_1"] = df.groupby("product_id")["quantity_sold"].shift(1)
    df["lag_7"] = df.groupby("product_id")["quantity_sold"].shift(7)
    df["lag_30"] = df.groupby("product_id")["quantity_sold"].shift(30)

    # ---- ROLLING ----
    df["rolling_7"] = (
        df.groupby("product_id")["quantity_sold"]
        .rolling(7).mean().reset_index(0, drop=True)
    )
    df["rolling_30"] = (
        df.groupby("product_id")["quantity_sold"]
        .rolling(30).mean().reset_index(0, drop=True)
    )

    # ---- DATE FEATURES ----
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ---- CATEGORY ENCODING ----
    df = pd.get_dummies(df, columns=["category"], drop_first=True)

    df = df.dropna().reset_index(drop=True)
    return df


# ============================
# BUILD SEQUENCES FOR LSTM
# ============================
def build_sequences(df, product_id, seq_len=30):
    df_p = df[df["product_id"] == product_id].copy()
    df_p = df_p.sort_values("date")

    if len(df_p) < seq_len:
        return None, None, None, None

    # LSTM features
    features = [
        'quantity_sold', 'lag_1', 'lag_7', 'lag_30',
        'rolling_7', 'rolling_30', 'is_festival',
        'is_weather_effecting', 'day_of_week', 'month'
    ]
    features += [c for c in df.columns if c.startswith("category_")]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_p[features])

    X, y = [], []

    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i][0])  # Only quantity_sold

    return np.array(X), np.array(y), scaler, df_p
