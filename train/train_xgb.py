import os, pandas as pd, numpy as np, joblib, xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

os.makedirs("artifacts", exist_ok=True)

# --- 读取数据：不依赖列名，直接取第一列 ---
df = pd.read_csv("data/WTI.csv", index_col=0, parse_dates=True)
s = df.iloc[:, 0].astype(float).dropna()

# --- 造特征：滞后 + 滚动统计 ---
def make_supervised(series: pd.Series, lags=20, roll_windows=(5,10,20)):
    df = pd.DataFrame({"y": series})
    for L in range(1, lags+1):
        df[f"lag{L}"] = series.shift(L)
    for R in roll_windows:
        df[f"rmean{R}"] = series.rolling(R).mean()
        df[f"rstd{R}"]  = series.rolling(R).std()
    df = df.dropna()
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    idx = df.index
    return X, y, idx

LAGS = 20
X, y, idx = make_supervised(s, lags=LAGS)

# --- 切分：最后30天为测试（与 ARIMA 对齐） ---
# 注意：特征行比原序列短，所以这里用最后 30 个样本
test_len = 30
split = len(X) - test_len
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]
idx_tr, idx_te = idx[:split], idx[split:]

# --- 训练 XGBoost ---
params = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "seed": 42,
}
dtr, dte = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xte, label=yte)
bst = xgb.train(params, dtr, num_boost_round=600)

# --- 预测 + 指标 ---
pred = bst.predict(dte)
mape = mean_absolute_percentage_error(yte, pred) * 100
mae  = mean_absolute_error(yte, pred)
print(f"[XGB] MAPE={mape:.2f}%, MAE={mae:.4f}")

# --- 保存模型与元信息 ---
bst.save_model("artifacts/xgb.json")
joblib.dump({"lags": LAGS, "roll_windows": (5,10,20)}, "artifacts/xgb_meta.pkl")

# --- 画图（与 ARIMA 类似） ---
plt.figure(figsize=(10,4))
# 训练尾部（只画最后 150 个点）
tail_n = 150
# 原始序列的可视化索引与 y、pred 对齐
s_tr_tail = pd.Series(ytr, index=idx_tr).tail(tail_n)
s_te = pd.Series(yte, index=idx_te)
s_pr = pd.Series(pred, index=idx_te)

s_tr_tail.plot(label="train", linewidth=1)
s_te.plot(label="test", linewidth=1)
s_pr.plot(label="forecast", linewidth=1.5)
plt.title("XGBoost Forecast — last 150 train pts + test")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/forecast_xgb.png", dpi=150)
plt.close()

