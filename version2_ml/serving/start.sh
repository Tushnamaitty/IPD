#!/bin/sh
# Container entrypoint: API in the background, Streamlit in the foreground.
# Both bind 0.0.0.0 inside the container; publish them on 127.0.0.1 on the host.
set -e
cd /app

if [ ! -f version2_ml/serving/models/model.joblib ]; then
    echo "model.joblib not found in version2_ml/serving/models; mount it or copy it in at build time" >&2
    exit 1
fi

python -m uvicorn version2_ml.serving.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT INT TERM

export API_URL="http://localhost:8000"
exec python -m streamlit run version2_ml/serving/app_streamlit.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true
