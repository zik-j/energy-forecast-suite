import os, pandas as pd, joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

os.makedirs("artifacts", exist_ok=True)

# 读取CSV——直接取第一列为序列（不依赖列名）
df = pd.read_csv("data/WTI.csv", index_col=0, parse_dates=True)
s = df.iloc[:, 0].astype(float).dropna()

# 训练/测试划分（最后30天为测试）
train, test = s[:-30], s[-30:]

# 训练 ARIMA(1,1,1)
model = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

# 预测
pred = res.forecast(steps=len(test))

# 指标
mape = mean_absolute_percentage_error(test, pred) * 100
mae  = mean_absolute_error(test, pred)
print(f"MAPE={mape:.2f}%, MAE={mae:.4f}")

# 保存模型
joblib.dump(res, "artifacts/arima.pkl")

# -------- 绘图：forecast.png --------
plt.figure(figsize=(10,4))
# 为了看得清，训练集只画最后 150 个点
train.tail(150).plot(label="train", linewidth=1)
test.plot(label="test", linewidth=1)
pred.index = test.index  # 对齐索引（statsmodels 可能给整数索引）
pred.plot(label="forecast", linewidth=1.5)
plt.title("ARIMA(1,1,1) Forecast — last 150 train pts + test")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/forecast.png", dpi=150)
plt.close()

# -------- 绘图：residuals.png --------
residuals = test - pred
plt.figure(figsize=(10,3.2))
residuals.plot()
plt.axhline(0, linestyle="--")
plt.title("Forecast Residuals (test - forecast)")
plt.xlabel("Date"); plt.ylabel("Error")
plt.tight_layout()
plt.savefig("artifacts/residuals.png", dpi=150)
plt.close()

