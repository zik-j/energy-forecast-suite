# streamlit_app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path

# --------------------
# Paths & constants
# --------------------
DATA_PATH = "data/WTI.csv"

ARIMA_PATH = "artifacts/arima.pkl"
XGB_JSON = "artifacts/xgb.json"
XGB_META = "artifacts/xgb_meta.pkl"

# --------------------
# Utilities
# --------------------
def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    """
    自动选择价格列：优先 Adj Close / Close；都没有就选最后一个数值列；
    仍不行，才选最后一列。
    """
    candidates = ["Adj Close", "Adj_Close", "adj close", "adj_close",
                  "Close", "close", "Price", "price"]
    for col in candidates:
        if col in df.columns:
            s = df[col].astype(float).dropna()
            if len(s) >= 5:
                return s

    # 没有候选列：尽量找最后一个数值列
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        s = df[numeric_cols[-1]].astype(float).dropna()
        if len(s) >= 5:
            return s

    # 最后兜底：最后一列
    s = df.iloc[:, -1].astype(float).dropna()
    if len(s) < 5:
        raise ValueError("Not enough numeric rows in data to build a series.")
    return s


@st.cache_data(show_spinner=False)
def read_price_series(path: str = DATA_PATH) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(how="all")
    return _pick_price_series(df)


@st.cache_resource(show_spinner=False)
def load_arima_model():
    if not os.path.exists(ARIMA_PATH):
        return None
    return joblib.load(ARIMA_PATH)


@st.cache_resource(show_spinner=False)
def load_xgb_model_and_meta():
    """
    返回 (xgb_model, xgb_booster, lags_list)
    - 兼容 meta 是 dict/list/int 等各种奇怪格式
    - 保证最终 lags 一定是 list[int]
    """
    try:
        import xgboost as xgb
    except Exception as e:
        st.error(f"Import xgboost failed: {e}")
        return None, None, [1, 2, 3, 4, 5]

    booster = xgb.Booster()
    if os.path.exists(XGB_JSON):
        booster.load_model(str(XGB_JSON))

    # 读 meta；不存在/格式不对就兜底
    if os.path.exists(XGB_META):
        meta_raw = joblib.load(XGB_META)
    else:
        meta_raw = {"lags": [1, 2, 3, 4, 5]}

    # 统一得到 raw_lags
    if isinstance(meta_raw, dict):
        raw_lags = meta_raw.get("lags", [1, 2, 3, 4, 5])
    else:
        raw_lags = meta_raw  # 可能本身就是 list 或 int

    # 规范化成 list[int]
    if isinstance(raw_lags, int):
        lags = list(range(1, int(raw_lags) + 1))
    elif hasattr(raw_lags, "__iter__"):
        lags = [int(l) for l in list(raw_lags)]
    else:
        lags = [int(raw_lags)]

    # 确保严格升序且去重
    lags = sorted(set(int(l) for l in lags if int(l) > 0))

    return xgb, booster, lags


def arima_forecast_multi(arima_model, steps: int) -> np.ndarray:
    """
    兼容 statsmodels 的 SARIMAXResults / ResultWrapper
    """
    try:
        res = arima_model.get_forecast(steps=steps)
        return res.predicted_mean.values
    except Exception:
        # 兜底：绝大多数情况下 .forecast 也能用
        return np.asarray(arima_model.forecast(steps=steps))


def xgb_forecast_multi(series: pd.Series,
                       xgb_mod,
                       xgb_booster,
                       lags: list[int],
                       horizon: int) -> np.ndarray:
    """
    递推方式做多步预测。保证 lags 已是 list[int]。
    """
    import xgboost as xgb
    tail = series.values.astype(float).copy()
    preds = []
    for _ in range(int(horizon)):
        feat = tail  # 用最近序列构造特征
        X = np.array([[feat[-l] for l in lags]], dtype=float)  # lags 一定是 list
        dtest = xgb.DMatrix(X)
        yhat = xgb_booster.predict(dtest)[0]
        preds.append(yhat)
        tail = np.append(tail, yhat)
    return np.array(preds)


def plot_forecast(st, hist: pd.Series, fcst: np.ndarray, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(len(hist)), hist.values, label="History")
    ax.plot(range(len(hist), len(hist) + len(fcst)), fcst, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


# --------------------
# UI
# --------------------
st.set_page_config(page_title="Energy Forecast Suite — Forecast", layout="wide")
st.title("🔮 Energy Forecast Suite — Forecast Dashboard")

with st.sidebar:
    st.header("⚙️ Controls")
    model_choice = st.selectbox("Model", ["arima", "xgb"])
    horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=14, step=1)
    last_n = st.slider("History points to show", min_value=100, max_value=600, value=200, step=50)
    st.caption("Tip: XGB uses recursive multi-step forecasting based on trained lags.")
    run_btn = st.button("Run Forecast", use_container_width=True)

# 读数据
try:
    s = read_price_series(DATA_PATH)
except Exception as e:
    st.error(f"Failed to read data: {e}")
    st.stop()

# 加载模型/元数据
arima_model = load_arima_model()
xgb_mod, xgb_booster, lags = load_xgb_model_and_meta()

# 运行 & 展示
if run_btn:
    hist = s.tail(last_n)
    if model_choice == "arima":
        if arima_model is None:
            st.error("ARIMA model not found: artifacts/arima.pkl")
            st.stop()
        fcst = arima_forecast_multi(arima_model, steps=horizon)
        plot_forecast(st, hist, fcst, f"ARIMA — last {last_n} history + {horizon} forecast")

        # 下载
        df_out = pd.DataFrame({"forecast": fcst})
        st.download_button(
            "Download forecast CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="forecast_arima.csv",
            mime="text/csv"
        )

    else:  # xgb
        if xgb_booster is None:
            st.error("XGB model not found or failed to load: artifacts/xgb.json / artifacts/xgb_meta.pkl")
            st.stop()
        fcst = xgb_forecast_multi(s, xgb_mod, xgb_booster, lags, horizon)
        plot_forecast(st, hist, fcst, f"XGB — last {last_n} history + {horizon} forecast (lags={lags})")

        # 下载
        df_out = pd.DataFrame({"forecast": fcst})
        st.download_button(
            "Download forecast CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="forecast_xgb.csv",
            mime="text/csv"
        )

