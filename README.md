# ENTWIN AI Service

MVP prediction service for **ENTWIN — Digital Life Twin**.

This is a FastAPI service. It provides lifestyle and sleep **risk indicators**, a planning **task-duration baseline**, and deterministic lifestyle **recommendations**. It is **not** a medical diagnostic system and it does **not** include an LLM assistant.

Version: `0.1.0`  
Default URL: [http://localhost:8090](http://localhost:8090)  
OpenAPI: [http://localhost:8090/docs](http://localhost:8090/docs)  
Health: [http://localhost:8090/health](http://localhost:8090/health)

The service never accesses Java microservice databases (`dlt_auth`, `dlt_planning`, `dlt_wellness`, `dlt_notification`). Java services will later send normalized JSON over HTTP (via the API Gateway).

---

## What it does

| Capability | Engine | Status |
| --- | --- | --- |
| Sleep risk indicator | Frozen `DecisionTreeClassifier` from the teammate project | Production |
| Lifestyle risk | Rule-based baseline (`evaluate_risk` + Wellness-unit mapping) | Production |
| Task duration | Deterministic `BASELINE_ESTIMATOR` | Contract foundation |
| Recommendations | Transparent threshold rules | MVP |
| AI assistant / LLM | — | Out of scope |

---

## Architecture

```text
Angular (later)
    → Spring API Gateway :8080   (JWT at the edge, later)
        → ENTWIN AI Service :8090
            POST /api/v1/ai/sleep-risk
            POST /api/v1/ai/lifestyle-risk
            POST /api/v1/ai/task-duration
            POST /api/v1/ai/recommendations
```

This milestone exposes a stable HTTP contract. It does **not** call Wellness or Planning over REST yet.

```text
ai/
├── app/
│   ├── main.py
│   ├── baseline.py                 # original teammate rule engine (preserved)
│   ├── api/routes/                 # health, models, sleep, lifestyle, planning, recommendations
│   ├── core/                       # config, logging, exceptions
│   ├── schemas/
│   ├── services/
│   ├── ml/                         # startup model loader + sleep preprocessing
│   └── models/                     # serialized .joblib + metadata JSON
├── scripts/                        # training / dataset generation (not used at runtime)
├── data/sleep_health_lifestyle.csv
├── notebooks/exploration.ipynb
├── tests/
├── Dockerfile
├── .env.example
└── requirements.txt
```

Models are loaded **once at application startup** (or fail startup if the required sleep model is missing). `joblib.load()` is not called per HTTP request.

---

## Endpoints

### Current (documented)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness. Does not reload models. |
| `GET` | `/api/v1/ai/health` | Same payload under the versioned prefix. |
| `GET` | `/api/v1/ai/models` | Public metadata for loaded models. No filesystem paths. |
| `POST` | `/api/v1/ai/sleep-risk` | Sleep risk indicator (ML). |
| `POST` | `/api/v1/ai/lifestyle-risk` | Lifestyle baseline from Wellness-shaped fields. |
| `POST` | `/api/v1/ai/task-duration` | Task duration baseline. |
| `POST` | `/api/v1/ai/recommendations` | Non-medical lifestyle suggestions. |

### Legacy aliases (deprecated, kept for teammate/manual tests)

The original service used **underscores**, not hyphens:

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/predict/sleep_disorder` | Original path. Same model, additive response fields. |
| `POST` | `/predict/sleep-disorder` | Hyphen alias. |
| `POST` | `/predict/lifestyle_risk` | Original path. Still accepts 0–100 domain scores. |
| `POST` | `/predict/lifestyle-risk` | Hyphen alias. |

Prefer `/api/v1/ai/...` for new work.

---

## Sleep-risk model

- **Algorithm:** `DecisionTreeClassifier` (`max_depth=5`, `random_state=42`)
- **Artifact:** `app/models/sleep_disorder_model.joblib` (preserved, not retrained by this refactor)
- **Feature order:** `app/models/sleep_disorder_features.joblib` is the single authority
- **Classes:** `Insomnia`, `None`, `Sleep Apnea` (academic dataset labels)
- **API framing:** informational **sleep risk indicator**, not a diagnosis

Exact feature order:

```text
Age
Sleep Duration
Quality of Sleep
Physical Activity Level
Stress Level
Heart Rate
Daily Steps
Gender          # Male=0, Female=1
BMI_Normal      # after mapping "Normal Weight" → "Normal"
BMI_Obese
BMI_Overweight
```

`predict_proba()` is exposed as `modelProbability`. That value is the tree’s vote share. It is **not** a calibrated clinical probability.

Example:

```json
{
  "age": 24,
  "gender": "Female",
  "sleepDuration": 7.2,
  "qualityOfSleep": 8,
  "physicalActivityLevel": 60,
  "stressLevel": 4,
  "bmiCategory": "Normal",
  "heartRate": 72,
  "dailySteps": 8000
}
```

Response:

```json
{
  "predictedClass": "None",
  "riskLevel": "LOW",
  "modelProbability": 0.82,
  "modelName": "sleep-risk",
  "modelVersion": "1.0.0",
  "disclaimer": "This result is an informational lifestyle risk indicator and is not a medical diagnosis."
}
```

Risk mapping: `None` → `LOW`, `Insomnia` → `MODERATE`, `Sleep Apnea` → `HIGH`.

---

## Lifestyle engine

`lifestyle_risk_model.joblib` **is a valid** `DecisionTreeClassifier`, trained on synthetic 0–100 scores (`sleep_score`, `hydration_score`, `activity_score`, `stress_score`) whose labels came from `evaluate_risk()` plus injected noise.

It is **not** used in production because:

1. It was trained to imitate the rule engine, not Wellness data.
2. Features are 0–100 scores, not minutes / ml / 1–10 mood scales.
3. Using it would falsely present a noisy copy of the rules as ML.

Production `POST /api/v1/ai/lifestyle-risk` uses `engine: "RULE_BASED_BASELINE"`.

Wellness-compatible request (all fields optional; missing values are skipped, **not** treated as 0):

```json
{
  "averageSleepMinutes": 420,
  "averageHydrationMl": 1700,
  "weeklyWorkoutMinutes": 180,
  "averageStress": 6.2,
  "averageFatigue": 5.4,
  "averageMood": 6.8,
  "averageDailySteps": 6200
}
```

Score mapping (higher = healthier, threshold 60):

| Input | Target used for 100 | Factor if below 60 |
| --- | --- | --- |
| `averageSleepMinutes` | 480 | `LOW_SLEEP` |
| `averageHydrationMl` | 2000 | `LOW_HYDRATION` |
| `weeklyWorkoutMinutes` | 150 | `LOW_ACTIVITY` |
| `averageDailySteps` | 8000 | `LOW_STEPS` |
| `averageStress` (1–10, higher is worse) | inverted | `HIGH_STRESS` |
| `averageFatigue` (1–10, higher is worse) | inverted | `HIGH_FATIGUE` |
| `averageMood` (1–10, higher is better) | `level * 10` | `LOW_MOOD` |

`riskLevel`: 0 factors → `LOW`; 1 factor → `MODERATE`; 2+ → `HIGH`.

The original 0–100 score endpoint is unchanged on the legacy paths.

---

## Task-duration baseline

No task-duration dataset or model exists in this repository.

`POST /api/v1/ai/task-duration` is a labeled **`BASELINE_ESTIMATOR`**:

- Blend user estimate (60%) and historical average (40%) when both are present
- Otherwise use the remaining estimate, or a category default
- Apply complexity (`EASY` 0.8 / `MEDIUM` 1.0 / `HARD` 1.3) and energy (`LOW` 0.9 / `MEDIUM` 1.0 / `HIGH` 1.15)
- When an estimate exists, multipliers are applied more mildly so user input is not overwritten
- **Priority is ignored** (it is not duration)

`confidence` is always `null` until a trained model exists.

---

## Recommendation rules

Deterministic, non-medical. Wording uses “your recent pattern shows…”, “you may want to…”, “consider…”.

| Signal | Threshold | Type / priority |
| --- | --- | --- |
| Hydration | `< 1500 ml` | `HYDRATION` / HIGH |
| Sleep | `< 360 min` | `REST` / HIGH |
| Sleep | `< 420 min` | `REST` / MEDIUM |
| Stress | `>= 7` | `STRESS` / HIGH |
| Fatigue | `>= 7` | `REST` / MEDIUM |
| Weekly workouts | `< 30 min` | `ACTIVITY` / HIGH |
| Weekly workouts | `< 75 min` | `ACTIVITY` / MEDIUM |
| Mood | `<= 4` | `MOOD` / MEDIUM |
| Steps | `< 4000` | `ACTIVITY` / MEDIUM |

Healthy values in a common range return an empty list (no alarming language).

---

## Dataset

File: `data/sleep_health_lifestyle.csv`

The teammate README identifies this as the public **Kaggle Sleep Health and Lifestyle Dataset**. Row count in this repo: **374**.

| | |
| --- | --- |
| Target | `Sleep Disorder` (`None` / `Insomnia` / `Sleep Apnea`; CSV nulls treated as `None`) |
| Rows | 374 |
| Class counts | None 219, Sleep Apnea 78, Insomnia 77 |
| Unused columns | `Person ID`, `Occupation`, `Blood Pressure` |

### Limitations

- Small academic/prototype dataset
- Class imbalance (`None` is the majority)
- Occupations and blood pressure are unused
- Duplicate-looking rows exist
- Not representative of ENTWIN users
- **Not a medical diagnostic tool**
- Holdout 96% is the original 80/20 split of this small table; 5-fold CV of equivalent trees is lower (~89% ± 4%)

If a later submission needs a formal citation, confirm the exact Kaggle dataset URL/version. It is not stored in the CSV itself.

---

## Model evaluation (existing production pickle)

Evaluated **without retraining**, using the original split (`test_size=0.2`, `random_state=42`, stratified):

| Metric | Value |
| --- | --- |
| Accuracy | 0.96 |
| Macro precision / recall / F1 | 0.9375 / 0.9494 / 0.9431 |
| Weighted precision / recall / F1 | 0.9617 / 0.9600 / 0.9606 |

Confusion matrix (`Insomnia`, `None`, `Sleep Apnea`):

```text
[[14,  0,  1]
 [ 1, 43,  0]
 [ 1,  0, 15]]
```

5-fold stratified CV of **new** trees with the same hyperparameters (not the frozen pickle): mean accuracy **0.8931**, std **0.0376**. This is not production validation.

Metadata: `app/models/sleep_disorder_model.metadata.json`

---

## Install and run

Python 3.12+ (local teammate venv may be 3.13). From `ai/`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

On macOS/Linux: `source venv/bin/activate` and `cp .env.example .env`.

Do not run training scripts at application startup.

---

## Tests

```bash
pytest
```

Regression samples captured from the original teammate model are in `tests/regression_samples.py`. The refactor must keep those sleep predictions unchanged.

---

## Retrain (optional, separate from the API)

```bash
python scripts/train_model.py --evaluate-only
python scripts/train_model.py --no-save
python scripts/train_model.py
python scripts/train_model.py --overwrite
```

Default training writes to `app/models/staging/` and does **not** replace the production pickle. Use `--overwrite` only after comparing metrics.

Synthetic lifestyle scores (for the unused lifestyle pickle experiment):

```bash
python scripts/generate_dataset.py
```

---

## Docker

```bash
docker build -t entwin-ai-service .
docker run --rm -p 8090:8090 entwin-ai-service
```

The image copies serialized models from `app/models/`. It does not train, and it does not download datasets.

---

## Configuration

See `.env.example`. Paths are resolved from the `ai/` project root, not from the process working directory. Do not set Windows-absolute paths in committed config.

| Variable | Default |
| --- | --- |
| `AI_SERVICE_HOST` | `0.0.0.0` |
| `AI_SERVICE_PORT` | `8090` |
| `AI_LOG_LEVEL` | `INFO` |
| `SLEEP_MODEL_PATH` | `app/models/sleep_disorder_model.joblib` |
| `SLEEP_FEATURES_PATH` | `app/models/sleep_disorder_features.joblib` |
| `LIFESTYLE_MODEL_PATH` | `app/models/lifestyle_risk_model.joblib` |

Logging records method, path, status, duration, and model name/version on predictions. It does **not** log JWT/Authorization, wellness notes, or full prediction bodies.

---

## Security (this milestone)

There is no second Auth Service in Python.

Intended path:

```text
Client → API Gateway (JWT) → AI Service
```

This MVP is backend-facing. It does not validate JWT yet and does not trust a client-supplied `userId`. Later, the AI service may also validate JWT for defense in depth. Gateway should not expose this service directly to the public internet.

---

## Future Java integration

### Wellness

Map `WeeklyWellnessSummaryResponse` fields into `POST /api/v1/ai/lifestyle-risk` and `POST /api/v1/ai/recommendations`. Do not give this service JDBC access to `dlt_wellness`.

### Planning

Map `plannedDurationMinutes`, `complexityLevel`, `energyRequired`, and later `actualDurationMinutes` into `POST /api/v1/ai/task-duration`. Collect real durations before training a model.

### API Gateway

A later Gateway route (not implemented in this milestone):

```text
/api/v1/ai/** → http://ai-service:8090
```

The AI app already serves `/api/v1/ai/**` so the Gateway can path-preserve like the other services.

---

## Out of scope

LLM assistant, RAG, embeddings, medical diagnosis/treatment, RabbitMQ/Kafka, Kubernetes, MLflow, automatic retraining, traffic prediction, AWS deployment.
