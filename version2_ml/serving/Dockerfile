# IPD V2 serving image: FastAPI (port 8000) + Streamlit (port 8501).
#
# Build from the repository root (the build context must be the repo root):
#   docker build -f version2_ml/serving/Dockerfile -t ipd-v2-serving .
#
# The exported model is NOT part of the source tree. Either mount it at run time
#   docker run --rm -p 127.0.0.1:8000:8000 -p 127.0.0.1:8501:8501 `
#     -v "${PWD}\version2_ml\serving\models:/app/version2_ml/serving/models:ro" ipd-v2-serving
# or run train_export.py before building so models/ is copied in.
# The repo-root .dockerignore is an allow-list: no NFHS/DHS data, outputs or
# virtual environments can enter the build context.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# xgboost needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY version2_ml/serving/requirements.txt /app/version2_ml/serving/requirements.txt
RUN pip install -r /app/version2_ml/serving/requirements.txt

# Inference code only: the serving package plus the research modules it imports.
COPY version2_ml/feature_builder.py version2_ml/model_framework.py version2_ml/preprocessing.py /app/version2_ml/
COPY version2_ml/experiments/A4/src /app/version2_ml/experiments/A4/src
COPY version2_ml/experiments/A7/src /app/version2_ml/experiments/A7/src
COPY version2_ml/serving /app/version2_ml/serving

RUN useradd --create-home --uid 1000 ipd && chmod +x /app/version2_ml/serving/start.sh
USER ipd

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["/app/version2_ml/serving/start.sh"]
