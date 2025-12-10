from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from utils import prepare_dataset
from model.predictor import forecast_product

# ============================
# FASTAPI APP SETUP
# ============================
app = FastAPI(title="RetailInsight Backend", version="1.0")

# Allow Streamlit (localhost:8501) to call this API
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# LOAD DATA ON STARTUP
# ============================
df = prepare_dataset()
PRODUCT_LIST: List[str] = sorted(df["product_id"].unique().tolist())


# ============================
# HEALTH CHECK
# ============================
@app.get("/health")
def health():
    return {"status": "ok", "products": len(PRODUCT_LIST)}


# ============================
# GET PRODUCT LIST
# ============================
@app.get("/products")
def get_products():
    """
    Return list of product IDs for the frontend dropdown.
    """
    return PRODUCT_LIST


# ============================
# FORECAST ENDPOINT
# ============================
@app.get("/predict/{product_id}")
def predict_product(product_id: str, horizon: int = 30):
    """
    LSTM forecast for given product_id and horizon (default 30 days).
    Returns:
      - history: last 120 days
      - forecast: next `horizon` days
    """
    if product_id not in PRODUCT_LIST:
        raise HTTPException(status_code=404, detail="Unknown product_id")

    result = forecast_product(product_id, horizon=horizon)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
