from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

MODEL_PATH = ARTIFACTS_DIR / "model_lgbm.txt"
FEATURE_STATE_PATH = ARTIFACTS_DIR / "feature_state.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Local eval mirrors the Nov–Dec holdout: train through Aug, hold out Sep–Oct.
LOCAL_TRAIN_END = "2025-08-31"
LOCAL_VAL_START = "2025-09-01"
FINAL_TRAIN_END = "2025-10-31"

RANDOM_SEED = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
