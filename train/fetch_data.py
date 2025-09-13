import os, yfinance as yf

os.makedirs("data", exist_ok=True)

# 下载 2 年日频的 WTI 原油
df = yf.download("CL=F", period="2y", interval="1d")

# 有 Adj Close 用它，否则用 Close
col = "Adj Close" if "Adj Close" in df.columns else "Close"
s = df[col].dropna()

# 存成：第一列为日期（索引），第二列为价格（只有这一列）
s.to_csv("data/WTI.csv")
print("saved data/WTI.csv, rows:", len(s), "col_name:", s.name)
PY

