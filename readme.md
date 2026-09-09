# Freight Rate Prediction

Predict truckload `posted_rate` from lane, equipment, weight, calendar, and market features.

Official brief: [`Freight_Rate_ML_Assessment.pdf`](Freight_Rate_ML_Assessment.pdf).

## Method

- **Split:** train on Jan–Aug 2025, evaluate on Sep–Oct (same temporal gap as the Nov–Dec holdout).
- **Target:** rate-per-mile (RPM); convert with `predicted_rate = rpm × distance`.
- **Model:** LightGBM on distance, equipment, calendar, geo, and market features.
- **Missing data:** weight by equipment median; market/quote by date → day-of-year → month.
- **December series:** fixed Lexington → Fort Wayne lane; coords and market signals filled from train-derived stats, then scored with the same model.

## Local results (Sep–Oct)

| Model | MAE | MAPE |
|-------|-----|------|
| Equipment median RPM | 229.10 | 10.49% |
| LightGBM | 112.34 | 4.97% |

Full metrics: [`artifacts/metrics.json`](artifacts/metrics.json).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Reproduce

```bash
python scripts/eda_summary.py   # optional
python -m src.train
python -m src.predict
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

## Outputs

| Path | Description |
|------|-------------|
| `validation_predictions.csv` | 12,000 rows: `load_id,predicted_rate` |
| `data/december_chart_inputs.csv` | December rates filled |
| `scorer_results/candidate_december.png` | Chart from `score.py` |
| `artifacts/` | Model, feature state, metrics |

## Layout

```
data/            train / validation / templates / december inputs
src/             load → features → train → predict
scripts/         eda_summary.py
artifacts/       saved model + metrics
scorer_results/  december chart
score.py         provided validator
report.pdf       write-up
```

## Report

Approach, data-quality notes, and the December chart are in [`report.pdf`](report.pdf) ([`report.md`](report.md) source).
