from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_load import load_december
from src.features import FeatureState

# Rough city centers if the train set somehow lacks these names.
_LEXINGTON = (38.0406, -84.5037)
_FORT_WAYNE = (41.0793, -85.1394)


def prepare_december_frame(state: FeatureState) -> pd.DataFrame:
    out = load_december().copy()
    plat, plon = state.city_coords.get("Lexington", _LEXINGTON)
    dlat, dlon = state.city_coords.get("Fort Wayne", _FORT_WAYNE)
    out["pickup_lat"], out["pickup_lon"] = plat, plon
    out["delivery_lat"], out["delivery_lon"] = dlat, dlon
    # Leave blank so transform_features applies seasonal market/quote fill.
    out["market_index"] = np.nan
    out["quote_signal"] = np.nan
    return out


def december_output_frame(original: pd.DataFrame, predicted_rate: np.ndarray) -> pd.DataFrame:
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    out = original.copy()
    out["predicted_rate"] = predicted_rate
    out = out[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out
