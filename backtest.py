# backtest.py
# Quant strategy prototype on top of your forecast models (ARIMA / XGB)
# - Turn predictions -> trading signals (long/short)
# - Backtest with PnL, equity curve
# - Metrics: Sharpe, Max Drawdown, Win Rate
# - Save CSV and plots to docs/strategy/

import os
import argparse
import math
import json
import requests
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

# -------- Project paths --------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WTI.csv"
ARTIFACTS = ROOT / "artifacts"
DOCS_STRATEGY_DIR = ROOT / "docs" / "strategy"
DOCS_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

ARIMA_PKL = ARTIFACTS / "arima.pkl"
XGB_JSON = ARTIFACTS / "xgb.json"
XGB_META = ARTIFACTS / "xgb_meta.pkl"

def read_price_series(path: Path = DATA_PATH) -> pd.Series:
    """
    Robust read: try Close/Adj Close... otherwise last column.
    Returns a float series with no NaNs.
    """
    df = pd.read_csv(path)
    # choose likely price columns
    for col in ["Close", "Adj Close", "Adj_Close", "close", "adj_close"]:
        if col in df.columns:
            s = df[col].astype(float).dropna()
            if len(s) >= 10:
                return s
    # fallback last column
    s = df.iloc[:, -1].astype(float).dropna()
    return s

# ---------- signal helpers ----------
def sma_sig(s: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    fast_ma = s.rolling(fast).mean()
    slow_ma = s.rolling(slow).mean()
    sig = np.sign(fast_ma - slow_ma)  # +1 long, -1 short
    return sig.fillna(0.0)

def ema_sig(s: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    fast_ma = s.ewm(span=fast, adjust=False).mean()
    slow_ma = s.ewm(span=slow, adjust=False).mean()
    sig = np.sign(fast_ma - slow_ma)
    return sig.fillna(0.0)

def from_forecast_to_signal(price: pd.Series, forecast: pd.Series, threshold: float = 0.0) -> pd.Series:
    """
    Align forecast with price tail; simple sign(forecast - last_price) rule on each point.
    If forecast shorter, align to the tail of price.
    """
    n = min(len(price), len(forecast))
    p = price.iloc[-n:].reset_index(drop=True)
    f = pd.Series(forecast.iloc[-n:].values, index=p.index)
    sig = np.sign((f - p) - threshold)
    return sig.fillna(0.0)

# ---------- metrics ----------
def equity_pnl_from_signal(price: pd.Series, signal: pd.Series, cost_bps: float = 5.0) -> Tuple[pd.DataFrame, dict]:
    """
    Very simple backtest: daily returns * signal (one-day lag to avoid lookahead).
    cost_bps is slippage/commission per trade (round trip approx).
    """
    n = min(len(price), len(signal))
    p = price.iloc[-n:].reset_index(drop=True)
    sig = signal.iloc[-n:].reset_index(drop=True)

    ret = p.pct_change().fillna(0.0)      # simple daily return
    # lag signal to avoid look-ahead (today's position earns next day's return)
    pos = sig.shift(1).fillna(0.0)

    # trading cost when position changes
    trade = pos.diff().abs().fillna(0.0)
    cost = trade * (cost_bps / 10000.0)

    strat_ret = pos * ret - cost
    equity = (1.0 + strat_ret).cumprod()

    # metrics
    sharpe = 0.0
    if strat_ret.std() > 1e-12:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min()

    win_rate = (strat_ret > 0).mean()
    avg_trade_pnl = strat_ret[trade > 0].mean() if (trade > 0).sum() > 0 else float(strat_ret.mean())

    pnl = pd.DataFrame({
        "price": p,
        "ret": ret,
        "signal": sig,
        "position": pos,
        "trade": trade,
        "cost": cost,
        "strat_ret": strat_ret,
        "equity": equity
    })

    metrics = {
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "avg_trade_pnl": round(float(avg_trade_pnl), 6)
    }
    return pnl, metrics

def save_outputs(pnl: pd.DataFrame, metrics: dict, tag: str = "run"):
    # CSV
    out_csv = DOCS_STRATEGY_DIR / f"pnl_{tag}.csv"
    pnl.to_csv(out_csv, index=False)

    # metrics json
    out_json = DOCS_STRATEGY_DIR / f"metrics_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # equity curve
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(pnl["equity"].values, label="Equity")
    ax[0].set_title(f"Equity Curve [{tag}]  Sharpe={metrics['sharpe']}  MaxDD={metrics['max_drawdown']}")
    ax[0].legend()

    ax[1].plot(pnl["price"].values, alpha=0.7, label="Price")
    ax[1].plot(pnl["position"].values, alpha=0.7, label="Position")
    ax[1].legend()

    out_png = DOCS_STRATEGY_DIR / f"equity_{tag}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_png}")

# ---------- main ----------
def run_backtest(args):
    # 1) price
    price = read_price_series(DATA_PATH)

    # 2) build signal by mode
    if args.mode == "forecast_csv":
        if not args.forecast_csv or not Path(args.forecast_csv).exists():
            raise FileNotFoundError("--forecast_csv is required and should exist")
        df = pd.read_csv(args.forecast_csv)
        # try common column names
        col = None
        for c in ["forecast", "yhat", "pred", "prediction"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            # if only one column exists, take it
            if df.shape[1] == 1:
                col = df.columns[0]
            else:
                raise ValueError("Could not find forecast column in CSV")
        forecast = df[col].astype(float)
        sig = from_forecast_to_signal(price, forecast, threshold=args.threshold)

    elif args.mode == "raw_price":
        # 使用简单 SMA/EMA 生成信号
        if args.signal_type == "sma":
            sig = sma_sig(price, fast=args.fast, slow=args.slow)
        else:
            sig = ema_sig(price, fast=args.fast, slow=args.slow)

    elif args.mode == "api":
        # 调用你的 FastAPI：/predict   {"model": "arima"|"xgb", "horizon": H}
        url = args.api_url
        payload = {"model": args.model, "horizon": int(args.horizon)}
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "forecast" not in data:
            raise ValueError("API response missing 'forecast'")
        forecast = pd.Series(data["forecast"], dtype=float)
        sig = from_forecast_to_signal(price, forecast, threshold=args.threshold)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 3) backtest
    pnl, metrics = equity_pnl_from_signal(price, sig, cost_bps=args.cost_bps)

    # 4) save
    tag = f"{args.mode}_{args.model}_{args.signal_type}"
    save_outputs(pnl, metrics, tag=tag)

def build_parser():
    p = argparse.ArgumentParser(description="Simple backtest on forecast -> signals -> PnL.")
    p.add_argument("--mode", type=str, default="forecast_csv",
                   choices=["forecast_csv", "raw_price", "api"],
                   help="Use forecast CSV / SMA/EMA on raw price / call FastAPI")
    p.add_argument("--forecast_csv", type=str, default="docs/samples/forecast_arima.csv",
                   help="CSV with column 'forecast'|'yhat'|'prediction'")
    p.add_argument("--model", type=str, default="arima", choices=["arima", "xgb"],
                   help="When mode=api, choose model")
    p.add_argument("--horizon", type=int, default=14,
                   help="steps ahead when calling API")

    p.add_argument("--signal_type", type=str, default="sma", choices=["sma", "ema"],
                   help="When mode=raw_price, choose signal type")
    p.add_argument("--fast", type=int, default=5)
    p.add_argument("--slow", type=int, default=20)

    p.add_argument("--threshold", type=float, default=0.0,
                   help="signal threshold for forecast mode")
    p.add_argument("--cost_bps", type=float, default=5.0,
                   help="round-trip cost in basis points")

    p.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/predict",
                   help="FastAPI endpoint for mode=api")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_backtest(args)
