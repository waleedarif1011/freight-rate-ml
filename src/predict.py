from __future__ import annotations

import joblib
import lightgbm as lgb
import pandas as pd

from src.config import (
    DECEMBER_PATH,
    FEATURE_STATE_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    TEMPLATE_PATH,
)
from src.data_load import load_validation
from src.december import december_output_frame, prepare_december_frame
from src.features import FeatureState, predict_rate, transform_features


def _load_artifacts() -> tuple[lgb.Booster, FeatureState]:
    if not MODEL_PATH.is_file() or not FEATURE_STATE_PATH.is_file():
        raise FileNotFoundError("missing artifacts — run `python -m src.train` first")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    state = FeatureState.from_dict(joblib.load(FEATURE_STATE_PATH))
    return model, state


def predict_validation(model: lgb.Booster, state: FeatureState) -> pd.DataFrame:
    val = load_validation()
    rates = predict_rate(model, transform_features(val, state)).round(2)
    preds = pd.DataFrame({"load_id": val["load_id"].astype(str), "predicted_rate": rates})

    template = pd.read_csv(TEMPLATE_PATH)
    out = template.drop(columns=["predicted_rate"], errors="ignore").merge(preds, on="load_id")
    if out["predicted_rate"].isna().any():
        raise ValueError("template has load_ids without predictions")
    return out[["load_id", "predicted_rate"]]


def predict_december(model: lgb.Booster, state: FeatureState) -> pd.DataFrame:
    prepared = prepare_december_frame(state)
    rates = predict_rate(model, transform_features(prepared, state)).round(2)
    return december_output_frame(prepared, rates)


def main() -> None:
    model, state = _load_artifacts()

    val_preds = predict_validation(model, state)
    val_preds.to_csv(PREDICTIONS_PATH, index=False)
    print(f"wrote {PREDICTIONS_PATH.name} ({len(val_preds):,} rows)")

    dec_preds = predict_december(model, state)
    dec_preds.to_csv(DECEMBER_PATH, index=False)
    print(
        f"wrote {DECEMBER_PATH.name} "
        f"min={dec_preds.predicted_rate.min():.2f} "
        f"med={dec_preds.predicted_rate.median():.2f} "
        f"max={dec_preds.predicted_rate.max():.2f}"
    )


if __name__ == "__main__":
    main()
