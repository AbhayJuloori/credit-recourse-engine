---
title: Credit Recourse Engine
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Credit Recourse Engine

> Most credit risk models tell you *who* will default. This project tells a denied applicant *what to change* — and ranks those changes by effort and success probability.

---

## The problem with SHAP

Every credit risk project ends at the same place:

```
XGBoost → SHAP bar chart → "low EXT_SOURCE_2 is your biggest risk factor"
```

The analyst knows *why*. The applicant still doesn't know *what to do*. The loan officer has a number, not a decision.

This project builds the missing layers.

---

## 4-Layer Pipeline

```
[1] XGBoost Classifier
    Trained on Home Credit Default Risk (307k rows, 300+ engineered features).
    Outputs P(default). Achieves 0.80+ AUC with full feature engineering.
    → xgboost, optuna (Bayesian search, not grid search)

        ↓

[2] Conformal Grey Zone — MAPIE RAPS
    Wraps the classifier in a conformal predictor.
    Each application gets: Approve / Grey Zone / Deny.
    Grey Zone = model confidence is too low to auto-deny. Treat differently.
    → mapie (RAPS method, 90% coverage)

        ↓

[3] Feasibility-Constrained Counterfactuals — DiCE-ML
    For grey-zone and denied applicants only.
    Generates recourse paths with:
      - Immutable features locked (age, gender, history)
      - Business constraints encoded (loan amount ±30%, DTI 0–60%)
      - Genetic algorithm for diverse paths (no differentiable model needed)
    → dice-ml (genetic backend)

        ↓

[4] Effort-Ranked Pathways — the decision layer
    Ranks counterfactual paths by:
      score = (flip_probability × 0.5) + (1 / (1 + effort) × 0.5)
    where:
      effort = Σ |Δfeature_i| / σ_i × time_weight_i

    Output:
      "Path 1 — Increase EXT_SOURCE_2 from 0.45 → 0.61 (~4 months) — 84% flip chance"
      "Path 2 — Reduce loan amount from $454k → $340k (immediately) — 79% flip chance"

    That is a decision, not a model.
```

---

## Key Design Choices

| Choice | What | Why |
|--------|------|-----|
| Bayesian HP search | Optuna TPE instead of grid search | More efficient, signals seniority |
| Conformal prediction | MAPIE RAPS at 90% coverage | Statistically valid uncertainty, better than softmax confidence |
| Genetic DiCE backend | `method='genetic'` | No differentiable model needed, better path diversity with XGBoost |
| Effort normalisation | Δ / σ × time_weight | Comparable across features with different scales and units |
| Imputation at inference | Training medians for missing features | Loan officer enters only key fields; rest imputed realistically |

---

## Stack

| Layer | Library | Version |
|-------|---------|---------|
| Base model | xgboost | 2.0.3 |
| Hyperparameter tuning | optuna | 3.5.0 |
| Conformal prediction | mapie | 0.8.3 |
| Counterfactuals | dice-ml | 0.11 |
| Explainability | shap | 0.44.0 |
| API | fastapi + uvicorn | 0.109.0 |
| Frontend | Tailwind + Alpine.js | CDN |

---

## Setup

### Prerequisites
- Python 3.9+
- Home Credit Default Risk dataset (download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk))

### Install

```bash
git clone https://github.com/<you>/credit-recourse-engine
cd credit-recourse-engine

make install          # creates .venv, installs all deps
source .venv/bin/activate
```

### Configure dataset path

Edit `backend/ml/config.py`:
```python
DATA_DIR = Path("/path/to/your/home-credit-default-risk")
```

### Train

```bash
make train
```

Expected: **90–180 minutes** on a modern laptop.
Expected AUC: **0.80–0.82** (with all supplementary tables).

Progress is logged to stdout:
```
[1/6] Building feature matrix…
[2/6] Encoding categorical features…
[3/6] Splitting data…
[4/6] Training XGBoost with Optuna… (60 trials)
[5/6] Calibrating MAPIE grey zone predictor…
[6/6] Setting up DiCE counterfactual generator…
```

### Run locally

```bash
make serve
# → http://localhost:8000
```

### Test

```bash
make test          # all tests
make test-fast     # constraints + ranker only (no ML artifacts needed)
```

---

## Feature Engineering

Six tables aggregated per applicant:

| Table | Rows | Features Added |
|-------|------|---------------|
| application_train | 307k | 122 raw + 20 engineered |
| bureau + bureau_balance | 1.7M + 27M | ~40 aggregations |
| previous_application | 1.7M | ~30 aggregations |
| POS_CASH_balance | 10M | ~15 aggregations |
| credit_card_balance | 3.8M | ~20 aggregations |
| installments_payments | 13.6M | ~25 aggregations |

**Total: ~300 features** — compared to 122 using application only.

---

## API Reference

### `POST /api/predict`

```json
{
  "AMT_INCOME_TOTAL": 112500,
  "AMT_CREDIT": 454500,
  "AMT_ANNUITY": 22500,
  "DAYS_BIRTH": -11680,
  "EXT_SOURCE_2": 0.52
}
```

Response:
```json
{
  "zone": "grey",
  "zone_label": "Grey Zone — Human Review",
  "zone_color": "#f59e0b",
  "p_default": 0.4821,
  "confidence": 0.5,
  "shap_top_features": [...]
}
```

### `POST /api/recourse`

```json
{
  "applicant": { ...same as predict... },
  "num_paths": 3
}
```

Response:
```json
{
  "zone": "grey",
  "p_default": 0.4821,
  "recourse_available": true,
  "paths": [
    {
      "rank": 1,
      "flip_probability": 0.84,
      "effort_score": 0.31,
      "composite_score": 0.74,
      "time_estimate": "~3–4 months",
      "steps": [
        {
          "action": "Increase External credit score 2 from 0.520 to 0.641 (Δ +0.121)",
          "time_estimate": "~3–5 months of good credit behaviour"
        }
      ]
    }
  ]
}
```

---

## Deployment (Free)

### Hugging Face Spaces

1. Train locally, commit artifacts to the repo (use Git LFS for `.pkl` files)
2. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) — **Docker** type
3. Push: the `Dockerfile` is already configured for HF Spaces (port 7860)
4. Get a permanent public URL — paste on LinkedIn

### Local network sharing

```bash
make serve-prod
# Share your local IP:8000 on your network
```

---

## Tests

```
backend/tests/
├── test_effort_ranker.py    # 15 tests — ranking logic, effort computation
├── test_constraints.py      # 16 tests — feature bounds, immutability, time weights
└── test_api.py              # 8 tests  — endpoint validation, error handling
```

Having tests at all puts you in the top 5% of DS portfolio projects.

---

## Project structure

```
credit-recourse-engine/
├── backend/
│   ├── ml/
│   │   ├── config.py              # paths, constants
│   │   ├── constraints.py         # immutable features, bounds, time weights
│   │   ├── feature_engineering.py # 6-table aggregation pipeline
│   │   ├── classifier.py          # XGBoost + Optuna
│   │   ├── grey_zone.py           # MAPIE conformal prediction
│   │   ├── counterfactuals.py     # DiCE-ML counterfactual generator
│   │   └── effort_ranker.py       # effort scoring + path ranking
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   └── routes/
│   │       ├── predict.py         # /api/predict
│   │       └── recourse.py        # /api/recourse
│   ├── scripts/
│   │   └── train.py               # full training pipeline
│   ├── artifacts/                  # saved models (gitignored)
│   └── tests/
├── frontend/
│   └── templates/index.html        # Tailwind + Alpine.js UI
├── Dockerfile                      # HuggingFace Spaces deployment
├── Makefile
├── requirements.txt
└── README.md
```

---

*Built on Home Credit Default Risk dataset. 307k applicants. 6 tables. 4 layers.*
