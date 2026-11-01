# IPD V2 serving: C-section probability API and demo UI

Packages the validated A7 model (XGBoost on `F1_detail_history`, Platt
calibration) as a saved model, a FastAPI service and a Streamlit demo.
Research demonstration only, not for clinical decisions; see
[model_card.md](model_card.md).

All commands are Windows PowerShell, run from the repository root
(`C:\...\IPD`).

## Layout

| File | Purpose |
| --- | --- |
| `train_export.py` | Refits the A7 model on the frozen split and writes `models/` |
| `artifacts.py`, `schemas.py`, `mapping.py`, `predictor.py`, `explain.py` | Inference core: request validation, request -> 12 model columns, calibrated prediction, per-request SHAP |
| `api.py` | FastAPI: `POST /predict`, `GET /health`, `GET /options`, `GET /model-card` |
| `app_streamlit.py` | Streamlit form; calls the API, falls back to in-process prediction |
| `tests/` | pytest suite (synthetic fixture model without NFHS data) |
| `model_card.md` | Intended use, data, metrics, limitations |
| `Dockerfile`, `start.sh` | Container running API (8000) and UI (8501) |
| `models/` | Exported artifacts; **gitignored, never committed** |

## 1. Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r version2_ml\serving\requirements.txt
```

Python 3.13 was used. `xgboost==3.4.1` and `scikit-learn==1.9.1` match the
A7 recording.

## 2. Train and export the model

Needs the local NFHS-5 birth recode `IABR7EFL.DTA` (not in the repository).

```powershell
$env:NFHS5_PATH = "C:\path\to\IABR7EFL.DTA"
.venv\Scripts\python.exe version2_ml\serving\train_export.py
# or: ... train_export.py --nfhs5-path C:\path\to\IABR7EFL.DTA --out-dir version2_ml\serving\models
```

This takes about 30 seconds. The script asserts the input sha256, the frozen
cohort (201,311 / 158,429 / 44,742), the split manifest hash and all
membership digests. It reproduces the A7 validation metrics and stops
without saving anything if ROC-AUC, PR-AUC, calibration slope or the
F1-optimal threshold fall outside their acceptance ranges. It never builds,
predicts or scores the test partition.

Outputs in `version2_ml\serving\models\`: `model.joblib`, `config.json`
(version, thresholds, risk bands, vocabulary, sha256), `metrics.json`,
`shap_summary.png`, `shap_top10.csv`. All are aggregate or model objects; no
row-level data. All are gitignored.

## 3. Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn version2_ml.serving.api:app --port 8000
```

Example request:

```powershell
$body = @{
  maternal_age_at_index_birth = 29; education_years = 12; wealth_index = 4
  birth_order = 2; twin_order = "single"
  facility_detail = "private: hospital/maternity home/clinic"
  residence = "urban"; religion = "1"
  is_first_birth = $false; previous_csection = "yes"
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
Invoke-RestMethod http://localhost:8000/health
```

`GET /options` lists every allowed value with labels. Invalid or inconsistent
inputs return HTTP 422 with a message. Request bodies are never logged, and
CORS allows localhost origins only. Interactive docs are at
http://localhost:8000/docs.

To use a model directory other than `version2_ml\serving\models`, set
`$env:IPD_MODEL_DIR`.

## 4. Run the UI

In a second terminal (start the API first, or the UI predicts in-process):

```powershell
.venv\Scripts\python.exe -m streamlit run version2_ml\serving\app_streamlit.py
```

Open http://localhost:8501. Set `$env:API_URL` if the API is not at
`http://localhost:8000`. The repo-root `.streamlit\config.toml` binds the UI
to localhost and disables Streamlit usage statistics. The UI stores nothing.

### Demo screenshots

`demo\demo_low.png`, `demo_medium.png` and `demo_high.png` show three invented
profiles (0.7% Low, 16.5% Medium, 95.2% High). To regenerate them with the API
and the UI running (Playwright is a dev-only tool, not in `requirements.txt`):

```powershell
.venv\Scripts\python.exe -m pip install playwright
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe version2_ml\serving\demo\capture_demo.py
```

The script checks that each probability shown in the UI matches a direct
`POST /predict` call.

## 5. Run the tests

```powershell
# without NFHS data: builds a synthetic fixture model in a temp folder
Remove-Item Env:NFHS5_PATH -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m pytest version2_ml\serving\tests -s

# with the real exported model, plus the golden reproduction test
$env:NFHS5_PATH = "C:\path\to\IABR7EFL.DTA"
.venv\Scripts\python.exe -m pytest version2_ml\serving\tests -s
```

The tests cover schema validation, request mapping, the golden metrics (real
model only), monotone direction of previous C-section, all endpoints, and
latency (200 sequential `/predict` calls; the p95 is printed and must be
under 300 ms).

The existing research suites are unchanged:

```powershell
.venv\Scripts\python.exe -m unittest discover -s version2_ml\tests -p "test_*.py"
foreach ($s in "common","A1","A2","A3","A4","A5","A6","A7") {
  $dir = if ($s -eq "common") { "version2_ml\experiments\common\tests" } else { "version2_ml\experiments\$s\tests" }
  .venv\Scripts\python.exe -m unittest discover -s $dir -p "test_*.py"
}
```

## 6. Docker

The image contains the serving code and the research modules it imports. The
repo-root `.dockerignore` is an allow-list, so NFHS/DHS data, `outputs\` and
`.venv\` never enter the build context. The model is not in git, so either
export it before building (then it is copied in) or mount it at run time.

```powershell
docker build -f version2_ml\serving\Dockerfile -t ipd-v2-serving .

docker run --rm -p 127.0.0.1:8000:8000 -p 127.0.0.1:8501:8501 `
  -v "${PWD}\version2_ml\serving\models:/app/version2_ml/serving/models:ro" `
  ipd-v2-serving
```

Publish on `127.0.0.1` as shown to keep both services local. The container
exits with a clear message if `models\model.joblib` is missing.

## Data protection

- Never commit NFHS data (`*.DTA`, parquet extracts), row-level predictions or
  `models\`. `.gitignore` covers these.
- `train_export.py` and the tests read validation rows only; the development
  test partition was scored once by A7 and is never re-scored here.
