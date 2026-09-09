from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_load import load_train, load_validation


def main() -> None:
    train = load_train()
    val = load_validation()

    print(f"train {train.shape}  {train['date'].min().date()} -> {train['date'].max().date()}")
    print(f"val   {val.shape}  {val['date'].min().date()} -> {val['date'].max().date()}")

    print("\nmissing (train)")
    print(train.isna().sum()[lambda s: s > 0].to_string())
    print("\nmissing (val)")
    print(val.isna().sum()[lambda s: s > 0].to_string())

    train = train.assign(rpm=train["posted_rate"] / train["distance"])
    print("\nequipment rpm")
    print(
        train.groupby("equipment")
        .agg(n=("rpm", "size"), mean_rpm=("rpm", "mean"), mean_rate=("posted_rate", "mean"))
        .round(3)
        .to_string()
    )

    print("\nmonthly rate / market")
    print(
        train.groupby(train["date"].dt.to_period("M"))
        .agg(
            mean_rate=("posted_rate", "mean"),
            mean_market=("market_index", "mean"),
            n=("posted_rate", "size"),
        )
        .round(3)
        .to_string()
    )

    lf = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne")]
    print(f"\nlexington -> fort wayne ({len(lf)})")
    if len(lf):
        print(lf["posted_rate"].describe()[["min", "50%", "max"]].round(2).to_dict())

    train_cities = set(train.pickup) | set(train.delivery)
    val_cities = set(val.pickup) | set(val.delivery)
    train_lanes = set(zip(train.pickup, train.delivery))
    val_lanes = set(zip(val.pickup, val.delivery))
    print(f"\nnew cities in val: {sorted(val_cities - train_cities)}")
    print(f"lane overlap: {len(val_lanes & train_lanes)} / {len(val_lanes)}")
    print(f"corr(distance, rate): {train.distance.corr(train.posted_rate):.4f}")


if __name__ == "__main__":
    main()
