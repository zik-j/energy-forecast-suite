# Energy Forecast Suite

A compact, production-ready demo for **time-series forecasting** on energy prices.  
It includes:
- **ARIMA baseline** (statsmodels)
- **XGBoost baseline** (recursive multi-step)
- **FastAPI** `/predict` endpoint
- **Streamlit** interactive dashboard
- **Docker** for both API & UI

---

## 1. Quick Start

### 1.1 Create & activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

