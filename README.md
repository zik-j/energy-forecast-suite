# Energy Forecast Suite

A compact, production‑ready demo for **time‑series forecasting** on energy prices.

**It includes:**

* **ARIMA** baseline (statsmodels)
* **XGBoost** baseline (recursive multi‑step)
* **FastAPI** `/predict` endpoint
* **Streamlit** interactive dashboard
* **Docker** support for API and UI

---

## 1. Quick Start

### 1.1 Create & activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Run FastAPI (backend API)

```bash
uvicorn app.api:app --reload
# Visit http://127.0.0.1:8000/docs
```

### 1.3 Run Streamlit (dashboard UI)

```bash
streamlit run streamlit_app.py --server.port 8501
# Visit http://127.0.0.1:8501
```

---

## 2. API Usage

**Example request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"arima","horizon":14}'
```

**Example response**

```json
{
  "model": "arima",
  "forecast": [69.2, 69.3, 69.1, 68.9, 69.0, 69.2, 69.4]
}
```

**Models supported**

```text
arima
xgb
```

---

## 3. Streamlit Dashboard

**Features**

```text
- Switch between models: ARIMA vs XGBoost
- Choose forecast horizon
- Display history & forecast curves
- Export results as CSV
```

**Sample results**

```text
docs/samples/forecast_arima.csv
docs/samples/forecast_xgb.csv
```

**Screenshots**

```text
docs/screenshots/streamlit_arima.png
docs/screenshots/streamlit_xgb.png
```

---

## 4. Docker Support

**Build image**

```bash
docker build -t energy-suite:latest --build-arg APP=streamlit .
```

**Run container**

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/artifacts":/app/artifacts:ro \
  -v "$PWD/data":/app/data:ro \
  energy-suite:latest
```

> If you want API only, build with `--build-arg APP=api` and expose port `8000`.

---

## 5. Project Structure

```text
energy-forecast-suite/
│
├── app/                 # FastAPI backend
│   └── api.py
├── artifacts/           # Pre-trained models (ARIMA, XGB)
├── data/                # Input datasets (WTI.csv)
├── docs/                # Docs, samples, screenshots
│   ├── samples/
│   │   ├── forecast_arima.csv
│   │   └── forecast_xgb.csv
│   └── screenshots/
│       ├── streamlit_arima.png
│       └── streamlit_xgb.png
├── tests/               # Test scripts
├── train/               # Training scripts
├── streamlit_app.py     # Streamlit dashboard
├── requirements.txt     # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

**Notes**

* Keep `artifacts/` in the repo so the demo models can load immediately.
* When running inside Docker on Linux, XGBoost often needs `libgomp1`. The provided `Dockerfile` installs it.
* If you see permission issues on mounted folders, check bind‑mount options on your OS.

