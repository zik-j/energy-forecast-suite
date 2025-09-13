# app/api.py
from fastapi import FastAPI
from pathlib import Path
import joblib, pandas as pd, numpy as np

import os

ARIMA_PATH = "artifacts/arima.pkl"
XGB_JSON = "artifacts/xgb.json"
XGB_META = "artifacts/xgb_meta.pkl"
DATA_PATH = "data/WTI.csv"

app = FastAPI(title="Energy Forecast API")

# ---------- utilities ----------
def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    candidates = ["Adj Close", "Adj_Close", "adj close", "adj_close",
                  "Close", "close", "Price", "price"]
    for col in candidates:
        if col in df.columns:
            s = df[col].astype(float).dropna()
            if len(s) >= 5:  # basic length check
                return s

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        s = df[numeric_cols[-1]].astype(float).dropna()
        if len(s) >= 5:
            return s
    s = df.iloc[:, -1].astype(float).dropna()
    if len(s) < 5:
        raise ValueError("No price-like column found and last column too short.")
    return s


def read_price_series(path: str = DATA_PATH) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(how="all")
    return _pick_price_series(df)


def load_arima():
    if not os.path.exists(ARIMA_PATH):
        return None
    return joblib.load(ARIMA_PATH)


def load_xgb():
    try:
        import xgboost as xgb
    except Exception:
        return None, None, [1, 2, 3, 4, 5]

    booster = xgb.Booster()
    if os.path.exists(XGB_JSON):
        booster.load_model(str(XGB_JSON))

    if os.path.exists(XGB_META):
        meta_raw = joblib.load(XGB_META)
    else:
        meta_raw = {"lags": [1, 2, 3, 4, 5]}

    if isinstance(meta_raw, dict):
        raw_lags = meta_raw.get("lags", [1, 2, 3, 4, 5])
    else:
        raw_lags = meta_raw

    if isinstance(raw_lags, int):
        lags = list(range(1, int(raw_lags) + 1))
    elif hasattr(raw_lags, "__iter__"):
        lags = [int(l) for l in list(raw_lags)]
    else:
        lags = [int(raw_lags)]

    lags = sorted(set(int(l) for l in lags if int(l) > 0))

    return xgb, booster, lags


def arima_forecast_multi(arima_model, steps: int) -> np.ndarray:
    try:
        res = arima_model.get_forecast(steps=steps)
        return res.predicted_mean.values
    except Exception:
        return np.asarray(arima_model.forecast(steps=steps))


def xgb_forecast_multi(series: pd.Series, xgb_mod, xgb_booster, lags: list[int], horizon: int) -> np.ndarray:
    import xgboost as xgb
    tail = series.values.astype(float).copy()
    preds = []
    for _ in range(int(horizon)):
        X = np.array([[tail[-l] for l in lags]], dtype=float)
        dtest = xgb.DMatrix(X)
        yhat = xgb_booster.predict(dtest)[0]
        preds.append(yhat)
        tail = np.append(tail, yhat)
    return np.array(preds)

# ---------- API ----------
@app.post("/predict")
def predict(request: dict):
    model = request.get("model", "arima")
    horizon = int(request.get("horizon", 14))

    s = read_price_series(DATA_PATH)

    if model == "arima":
        arima_model = load_arima()
        if arima_model is None:
            return {"error": "ARIMA model not found"}
        y = arima_forecast_multi(arima_model, steps=horizon).tolist()
        return {"model": "arima", "forecast": y}

    elif model == "xgb":
        xgb_mod, xgb_booster, lags = load_xgb()
        if xgb_booster is None:
            return {"error": "XGB model not found"}
        y = xgb_forecast_multi(s, xgb_mod, xgb_booster, lags, horizon).tolist()
        return {"model": "xgb", "forecast": y}

    else:
        return {"error": f"unknown model {model}"}

