from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    EARLY_STOPPING_ROUNDS,
    FEATURE_STATE_PATH,
    FINAL_TRAIN_END,
    LGBM_PARAMS,
    LOCAL_TRAIN_END,
    LOCAL_VAL_START,
    METRICS_PATH,
    MODEL_PATH,
    NUM_BOOST_ROUND,
)
from src.data_load import load_train
from src.features import (
    FEATURE_COLUMNS,
    baseline_rpm_predict,
    feature_matrix,
    fit_feature_state,
    predict_rate,
    transform_features,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)) * 100),
    }


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= LOCAL_TRAIN_END].copy()
    val = df[(df["date"] >= LOCAL_VAL_START) & (df["date"] <= FINAL_TRAIN_END)].copy()
    return train, val


def _fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_COLUMNS)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=100)]

    if x_val is not None and y_val is not None:
        valid_sets.append(
            lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=FEATURE_COLUMNS)
        )
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))

    return lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def train_pipeline() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    full = load_train()
    local_train, local_val = _time_split(full)
    print(f"split train={len(local_train):,}  val={len(local_val):,}")

    state = fit_feature_state(local_train)
    train_f = transform_features(local_train, state)
    val_f = transform_features(local_val, state)

    baseline = regression_metrics(
        local_val["posted_rate"].to_numpy(),
        baseline_rpm_predict(val_f, state),
    )
    print("baseline", baseline)

    model = _fit_lgbm(
        feature_matrix(train_f),
        train_f["rpm"].to_numpy(),
        feature_matrix(val_f),
        val_f["rpm"].to_numpy(),
    )
    gbm = regression_metrics(
        local_val["posted_rate"].to_numpy(),
        predict_rate(model, val_f, num_iteration=model.best_iteration),
    )
    print("lightgbm", gbm, f"best_iter={model.best_iteration}")

    # Refit on all labeled rows with the early-stopped round count.
    rounds = max(int(model.best_iteration or 400), 50)
    final_state = fit_feature_state(full)
    full_f = transform_features(full, final_state)
    final = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(
            feature_matrix(full_f),
            label=full_f["rpm"].to_numpy(),
            feature_name=FEATURE_COLUMNS,
        ),
        num_boost_round=rounds,
    )
    final.save_model(str(MODEL_PATH))
    joblib.dump(final_state.to_dict(), FEATURE_STATE_PATH)

    metrics = {
        "local_train_end": LOCAL_TRAIN_END,
        "local_val_start": LOCAL_VAL_START,
        "local_train_rows": int(len(local_train)),
        "local_val_rows": int(len(local_val)),
        "baseline": baseline,
        "lightgbm": gbm,
        "best_iteration_local": int(model.best_iteration or rounds),
        "final_num_boost_round": rounds,
        "feature_columns": FEATURE_COLUMNS,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {MODEL_PATH.name}, {FEATURE_STATE_PATH.name}, {METRICS_PATH.name}")
    return metrics


def main() -> None:
    train_pipeline()


if __name__ == "__main__":
    main()
