from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "distance",
    "log_distance",
    "weight",
    "equipment_code",
    "month",
    "dayofweek",
    "weekofyear",
    "month_sin",
    "month_cos",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_miles",
    "bearing",
    "market_index",
    "quote_signal",
    "distance_x_market",
]

EQUIPMENT_ORDER = ["Dry Van", "Reefer", "Flatbed"]


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def _median_map(series: pd.Series) -> dict:
    return {k: float(v) for k, v in series.dropna().items()}


@dataclass
class FeatureState:
    weight_median_by_equipment: dict[str, float] = field(default_factory=dict)
    global_weight_median: float = 0.0
    market_by_date: dict[str, float] = field(default_factory=dict)
    quote_by_date: dict[str, float] = field(default_factory=dict)
    global_market_median: float = 1.0
    global_quote_median: float = 2.0
    market_by_month: dict[int, float] = field(default_factory=dict)
    quote_by_month: dict[int, float] = field(default_factory=dict)
    market_by_dayofyear: dict[int, float] = field(default_factory=dict)
    quote_by_dayofyear: dict[int, float] = field(default_factory=dict)
    city_coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    equipment_to_code: dict[str, int] = field(default_factory=dict)
    equipment_rpm: dict[str, float] = field(default_factory=dict)
    global_rpm: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # joblib is fine with int keys, but keep dayofyear keys as str for stability
        payload["market_by_dayofyear"] = {str(k): v for k, v in self.market_by_dayofyear.items()}
        payload["quote_by_dayofyear"] = {str(k): v for k, v in self.quote_by_dayofyear.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureState:
        data = dict(payload)
        data["market_by_month"] = {int(k): float(v) for k, v in data["market_by_month"].items()}
        data["quote_by_month"] = {int(k): float(v) for k, v in data["quote_by_month"].items()}
        data["market_by_dayofyear"] = {
            int(k): float(v) for k, v in data["market_by_dayofyear"].items()
        }
        data["quote_by_dayofyear"] = {
            int(k): float(v) for k, v in data["quote_by_dayofyear"].items()
        }
        data["city_coords"] = {
            k: (float(v[0]), float(v[1])) for k, v in data["city_coords"].items()
        }
        return cls(**data)


def fit_feature_state(train: pd.DataFrame) -> FeatureState:
    state = FeatureState()
    state.equipment_to_code = {name: i for i, name in enumerate(EQUIPMENT_ORDER)}

    state.global_weight_median = float(train["weight"].median(skipna=True))
    for equip, g in train.groupby("equipment"):
        med = g["weight"].median(skipna=True)
        if pd.notna(med):
            state.weight_median_by_equipment[str(equip)] = float(med)

    state.global_market_median = float(train["market_index"].median(skipna=True))
    state.global_quote_median = float(train["quote_signal"].median(skipna=True))

    dated = train.assign(_d=train["date"].dt.strftime("%Y-%m-%d"))
    state.market_by_date = _median_map(dated.groupby("_d")["market_index"].median())
    state.quote_by_date = _median_map(dated.groupby("_d")["quote_signal"].median())
    state.market_by_month = {
        int(k): float(v)
        for k, v in train.groupby(train["date"].dt.month)["market_index"].median().dropna().items()
    }
    state.quote_by_month = {
        int(k): float(v)
        for k, v in train.groupby(train["date"].dt.month)["quote_signal"].median().dropna().items()
    }
    state.market_by_dayofyear = {
        int(k): float(v)
        for k, v in train.groupby(train["date"].dt.dayofyear)["market_index"]
        .median()
        .dropna()
        .items()
    }
    state.quote_by_dayofyear = {
        int(k): float(v)
        for k, v in train.groupby(train["date"].dt.dayofyear)["quote_signal"]
        .median()
        .dropna()
        .items()
    }

    pickup = (
        train.dropna(subset=["pickup_lat", "pickup_lon"])
        .groupby("pickup")[["pickup_lat", "pickup_lon"]]
        .median()
    )
    delivery = (
        train.dropna(subset=["delivery_lat", "delivery_lon"])
        .groupby("delivery")[["delivery_lat", "delivery_lon"]]
        .median()
    )
    coords: dict[str, tuple[float, float]] = {}
    for city, row in pickup.iterrows():
        coords[str(city)] = (float(row.pickup_lat), float(row.pickup_lon))
    for city, row in delivery.iterrows():
        key = str(city)
        lat, lon = float(row.delivery_lat), float(row.delivery_lon)
        if key in coords:
            plat, plon = coords[key]
            coords[key] = ((plat + lat) / 2, (plon + lon) / 2)
        else:
            coords[key] = (lat, lon)
    state.city_coords = coords

    rpm = train["posted_rate"] / train["distance"].replace(0, np.nan)
    state.global_rpm = float(rpm.median(skipna=True))
    for equip, g in train.groupby("equipment"):
        med = (g["posted_rate"] / g["distance"].replace(0, np.nan)).median(skipna=True)
        if pd.notna(med):
            state.equipment_rpm[str(equip)] = float(med)

    return state


def _impute_weight(df: pd.DataFrame, state: FeatureState) -> pd.Series:
    weight = df["weight"].astype(float).copy()
    miss = weight.isna()
    if miss.any():
        filled = df.loc[miss, "equipment"].map(state.weight_median_by_equipment)
        weight.loc[miss] = filled.fillna(state.global_weight_median)
    return weight


def _fill_series(
    series: pd.Series,
    date_keys: pd.Series,
    months: pd.Series,
    doy: pd.Series,
    by_date: dict,
    by_doy: dict,
    by_month: dict,
    fallback: float,
) -> pd.Series:
    out = series.copy()
    for mapping, keys in (
        (by_date, date_keys),
        (by_doy, doy),
        (by_month, months),
    ):
        miss = out.isna()
        if miss.any():
            out.loc[miss] = keys.loc[miss].map(mapping)
    return out.fillna(fallback).astype(float)


def _impute_market_quote(df: pd.DataFrame, state: FeatureState) -> tuple[pd.Series, pd.Series]:
    idx = df.index
    market = (
        df["market_index"].astype(float).copy()
        if "market_index" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    quote = (
        df["quote_signal"].astype(float).copy()
        if "quote_signal" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    date_keys = df["date"].dt.strftime("%Y-%m-%d")
    months = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    market = _fill_series(
        market,
        date_keys,
        months,
        doy,
        state.market_by_date,
        state.market_by_dayofyear,
        state.market_by_month,
        state.global_market_median,
    )
    quote = _fill_series(
        quote,
        date_keys,
        months,
        doy,
        state.quote_by_date,
        state.quote_by_dayofyear,
        state.quote_by_month,
        state.global_quote_median,
    )
    return market, quote


def _fill_coords(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    out = df.copy()
    fallback_lat = float(np.median([c[0] for c in state.city_coords.values()])) if state.city_coords else 39.0
    fallback_lon = float(np.median([c[1] for c in state.city_coords.values()])) if state.city_coords else -90.0

    for city_col, lat_col, lon_col in (
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ):
        if lat_col not in out.columns:
            out[lat_col] = np.nan
        if lon_col not in out.columns:
            out[lon_col] = np.nan

        miss = out[lat_col].isna() | out[lon_col].isna()
        if miss.any():
            mapped = out.loc[miss, city_col].map(state.city_coords)
            out.loc[miss, lat_col] = mapped.map(lambda c: c[0] if isinstance(c, tuple) else np.nan)
            out.loc[miss, lon_col] = mapped.map(lambda c: c[1] if isinstance(c, tuple) else np.nan)

        out[lat_col] = out[lat_col].fillna(fallback_lat)
        out[lon_col] = out[lon_col].fillna(fallback_lon)
    return out


def transform_features(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    out = _fill_coords(df, state).copy()
    out["weight"] = _impute_weight(out, state)
    out["market_index"], out["quote_signal"] = _impute_market_quote(out, state)

    dist = out["distance"].astype(float)
    out["log_distance"] = np.log1p(dist)
    out["equipment_code"] = out["equipment"].map(state.equipment_to_code).fillna(-1).astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["dayofweek"] = out["date"].dt.dayofweek.astype(int)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    plat, plon = out["pickup_lat"].to_numpy(), out["pickup_lon"].to_numpy()
    dlat, dlon = out["delivery_lat"].to_numpy(), out["delivery_lon"].to_numpy()
    out["haversine_miles"] = _haversine_miles(plat, plon, dlat, dlon)
    out["bearing"] = _bearing(plat, plon, dlat, dlon)
    out["distance_x_market"] = dist * out["market_index"].astype(float)

    if "posted_rate" in out.columns:
        out["rpm"] = out["posted_rate"] / dist.replace(0, np.nan)
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].astype(float)


def baseline_rpm_predict(df: pd.DataFrame, state: FeatureState) -> np.ndarray:
    rpm = df["equipment"].map(state.equipment_rpm).fillna(state.global_rpm).astype(float)
    return (rpm * df["distance"].astype(float)).to_numpy()


def predict_rate(model, df: pd.DataFrame, *, num_iteration: int | None = None) -> np.ndarray:
    kwargs = {}
    if num_iteration is not None:
        kwargs["num_iteration"] = num_iteration
    rpm = model.predict(feature_matrix(df), **kwargs)
    return np.clip(np.asarray(rpm) * df["distance"].astype(float).to_numpy(), 1e-3, None)
