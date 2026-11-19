"""Smoke tests for feature engineering, model inference, and score helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from score import EXPECTED_ROWS, numeric_series, validate_december
from src.config import FEATURE_STATE_PATH, MODEL_PATH
from src.features import (
    FEATURE_COLUMNS,
    FeatureState,
    fit_feature_state,
    predict_rate,
    transform_features,
)
from src.train import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pickup": "Lexington",
                "delivery": "Atlanta",
                "pickup_lat": 36.99,
                "pickup_lon": -85.0,
                "delivery_lat": 34.85,
                "delivery_lon": -86.29,
                "distance": 209.1,
                "equipment": "Dry Van",
                "weight": 30000.0,
                "date": pd.Timestamp("2025-09-15"),
                "market_index": 1.1,
                "quote_signal": 2.2,
                "posted_rate": 500.0,
            },
            {
                "pickup": "Phoenix",
                "delivery": "Shreveport",
                "pickup_lat": 28.76,
                "pickup_lon": -115.88,
                "delivery_lat": 32.55,
                "delivery_lon": -92.32,
                "distance": 1669.0,
                "equipment": "Reefer",
                "weight": np.nan,
                "date": pd.Timestamp("2025-10-01"),
                "market_index": np.nan,
                "quote_signal": np.nan,
                "posted_rate": 3200.0,
            },
        ]
    )


def test_feature_engineering_shape_and_columns():
    raw = _tiny_frame()
    state = fit_feature_state(raw.dropna(subset=["weight", "market_index"]))
    # Fit needs at least one complete row; keep full frame for transform.
    feats = transform_features(raw, state)
    matrix = feats[FEATURE_COLUMNS]
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix.shape == (len(raw), len(FEATURE_COLUMNS))
    assert matrix.isna().sum().sum() == 0
    assert (feats["weight"] > 0).all()


def test_model_load_and_single_row_prediction():
    assert MODEL_PATH.is_file(), "artifacts/model_lgbm.txt missing"
    assert FEATURE_STATE_PATH.is_file(), "artifacts/feature_state.joblib missing"

    model = lgb.Booster(model_file=str(MODEL_PATH))
    state = FeatureState.from_dict(joblib.load(FEATURE_STATE_PATH))
    row = pd.DataFrame(
        [
            {
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "distance": 360.0,
                "equipment": "Dry Van",
                "weight": 32000.0,
                "date": pd.Timestamp("2025-12-15"),
                "market_index": np.nan,
                "quote_signal": np.nan,
                "pickup_lat": np.nan,
                "pickup_lon": np.nan,
                "delivery_lat": np.nan,
                "delivery_lon": np.nan,
            }
        ]
    )
    rate = float(predict_rate(model, transform_features(row, state))[0])
    assert np.isfinite(rate)
    assert rate > 0


def test_regression_metrics_sanity():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx((10 + 10 + 30) / 3)
    assert metrics["mape"] == pytest.approx((10 / 100 + 10 / 200 + 30 / 300) / 3 * 100)
    assert metrics["rmse"] > 0


def test_score_helpers_sanity():
    assert EXPECTED_ROWS == 12_000
    frame = pd.DataFrame({"predicted_rate": ["10.5", "20"]})
    series = numeric_series(frame, "predicted_rate", "test")
    assert series.tolist() == [10.5, 20.0]

    december = pd.DataFrame(
        {
            "pickup": ["Lexington"] * 31,
            "delivery": ["Fort Wayne"] * 31,
            "distance": [360.0] * 31,
            "equipment": ["Dry Van"] * 31,
            "weight": [32_000.0] * 31,
            "date": pd.date_range("2025-12-01", "2025-12-31", freq="D").strftime("%Y-%m-%d"),
            "predicted_rate": np.linspace(700.0, 800.0, 31),
        }
    )
    validated = validate_december(december)
    assert len(validated) == 31
    assert validated["predicted_rate"].min() > 0
