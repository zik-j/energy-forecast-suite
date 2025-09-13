# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Debian 包：libgomp1 是 XGBoost 运行时依赖；curl 用于 healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（提高缓存命中）
COPY requirements.txt .
RUN pip install -r requirements.txt

# 再拷贝代码
COPY . .

# 可切换运行目标：api / streamlit
ARG APP=streamlit
ENV APP=${APP}

# 暴露两个端口，按需使用
EXPOSE 8000 8501

# 健康检查：API 走 8000，Streamlit 走 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD \
    bash -lc 'if [ "$APP" = "api" ]; then curl -fsS http://localhost:8000/docs >/dev/null || exit 1; else curl -fsS http://localhost:8501/_stcore/health >/dev/null || exit 1; fi'

# 根据 APP 选择启动命令
CMD bash -lc 'if [ "$APP" = "api" ]; then \
      exec uvicorn app.api:app --host 0.0.0.0 --port 8000; \
    else \
      exec streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0; \
    fi'

