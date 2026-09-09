from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DECEMBER_PATH, TRAIN_PATH, VALIDATION_PATH

_GEO_MARKET = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("invalid dates in input")
    return out


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_train(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or TRAIN_PATH)
    return _to_numeric(_parse_dates(df), _GEO_MARKET + ["posted_rate"])


def load_validation(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or VALIDATION_PATH)
    return _to_numeric(_parse_dates(df), _GEO_MARKET)


def load_december(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DECEMBER_PATH)
    return _to_numeric(_parse_dates(df), ["distance", "weight"])
