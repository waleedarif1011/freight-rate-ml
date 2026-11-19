"""Gradio demo: quote a truckload lane with the saved LightGBM model."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import lightgbm as lgb
import pandas as pd

from src.config import FEATURE_STATE_PATH, METRICS_PATH, MODEL_PATH, ROOT
from src.features import EQUIPMENT_ORDER, FeatureState, predict_rate, transform_features

LANES_PATH = ROOT / "artifacts" / "lanes.json"
DECEMBER_CHART = ROOT / "scorer_results" / "candidate_december.png"

PRESETS = [
    {
        "label": "Lexington → Atlanta (short Dry Van)",
        "pickup": "Lexington",
        "delivery": "Atlanta",
        "equipment": "Dry Van",
        "weight": 29262.0,
        "month": 9,
        "day": 15,
        "distance": 209.1,
        "market_index": 1.13,
    },
    {
        "label": "Phoenix → Shreveport (long Dry Van)",
        "pickup": "Phoenix",
        "delivery": "Shreveport",
        "equipment": "Dry Van",
        "weight": 31821.0,
        "month": 10,
        "day": 5,
        "distance": 1669.0,
        "market_index": 1.13,
    },
    {
        "label": "Columbia → Oklahoma City (mid Flatbed)",
        "pickup": "Columbia",
        "delivery": "Oklahoma City",
        "equipment": "Flatbed",
        "weight": 32448.0,
        "month": 9,
        "day": 22,
        "distance": 1058.9,
        "market_index": 1.04,
    },
]


def _load_artifacts() -> tuple[lgb.Booster, FeatureState, dict, list[dict]]:
    if not MODEL_PATH.is_file() or not FEATURE_STATE_PATH.is_file():
        raise FileNotFoundError("missing artifacts — run `python -m src.train` first")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    state = FeatureState.from_dict(joblib.load(FEATURE_STATE_PATH))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.is_file() else {}
    lanes = json.loads(LANES_PATH.read_text(encoding="utf-8")) if LANES_PATH.is_file() else []
    return model, state, metrics, lanes


MODEL, STATE, METRICS, LANES = _load_artifacts()
LANE_CHOICES = [f"{row['pickup']} → {row['delivery']}" for row in LANES]
LANE_BY_LABEL = {label: row for label, row in zip(LANE_CHOICES, LANES)}


def metrics_markdown() -> str:
    baseline = METRICS.get("baseline", {})
    lightgbm = METRICS.get("lightgbm", {})
    return (
        "### Model metrics (Sep–Oct holdout)\n\n"
        "| Model | MAE ($) | MAPE |\n"
        "|-------|---------|------|\n"
        f"| Equipment median RPM | {baseline.get('mae', float('nan')):.2f} | "
        f"{baseline.get('mape', float('nan')):.2f}% |\n"
        f"| **LightGBM** | **{lightgbm.get('mae', float('nan')):.2f}** | "
        f"**{lightgbm.get('mape', float('nan')):.2f}%** |\n"
    )


def quote_lane(
    lane: str,
    equipment: str,
    weight: float,
    month: int,
    day: int,
    distance: float,
    market_index: float,
) -> tuple[str, str, str, str | Path | None]:
    meta = LANE_BY_LABEL.get(lane)
    if meta is None:
        return "Unknown lane", "", metrics_markdown(), DECEMBER_CHART if DECEMBER_CHART.is_file() else None

    pickup, delivery = meta["pickup"], meta["delivery"]
    year = 2025
    # Clamp day to a safe calendar day for the chosen month.
    day = int(max(1, min(int(day), 28)))
    row = pd.DataFrame(
        [
            {
                "pickup": pickup,
                "delivery": delivery,
                "distance": float(distance),
                "equipment": equipment,
                "weight": float(weight),
                "date": pd.Timestamp(year=year, month=int(month), day=day),
                "market_index": float(market_index),
                "quote_signal": float("nan"),
                "pickup_lat": float("nan"),
                "pickup_lon": float("nan"),
                "delivery_lat": float("nan"),
                "delivery_lon": float("nan"),
            }
        ]
    )
    feats = transform_features(row, STATE)
    rate = float(predict_rate(MODEL, feats)[0])
    rpm = rate / max(float(distance), 1e-6)
    summary = (
        f"**Predicted posted rate:** ${rate:,.2f}\n\n"
        f"**Rate per mile:** ${rpm:.3f}/mi\n\n"
        f"**Lane:** {pickup} → {delivery} · {equipment} · {distance:.0f} mi · "
        f"{year}-{int(month):02d}-{day:02d}"
    )
    chart = DECEMBER_CHART if DECEMBER_CHART.is_file() else None
    return f"${rate:,.2f}", f"${rpm:.3f}/mi", summary + "\n\n" + metrics_markdown(), chart


def apply_preset(preset_label: str):
    preset = next((p for p in PRESETS if p["label"] == preset_label), PRESETS[0])
    lane = f"{preset['pickup']} → {preset['delivery']}"
    return (
        lane,
        preset["equipment"],
        preset["weight"],
        preset["month"],
        preset["day"],
        preset["distance"],
        preset["market_index"],
    )


def build_app() -> gr.Blocks:
    default = apply_preset(PRESETS[0]["label"])
    with gr.Blocks(title="Freight Rate ML — Quote a Truckload Lane") as demo:
        gr.Markdown(
            "# Quote a Truckload Lane\n"
            "Spot `posted_rate` prediction from a LightGBM RPM model "
            "(distance, equipment, calendar, geo, market features)."
        )
        with gr.Row():
            with gr.Column():
                preset = gr.Dropdown(
                    choices=[p["label"] for p in PRESETS],
                    value=PRESETS[0]["label"],
                    label="Preset examples",
                )
                lane = gr.Dropdown(choices=LANE_CHOICES, value=default[0], label="Lane")
                equipment = gr.Dropdown(
                    choices=EQUIPMENT_ORDER, value=default[1], label="Equipment"
                )
                weight = gr.Number(value=default[2], label="Weight (lb)")
                month = gr.Slider(1, 12, value=default[3], step=1, label="Pickup month")
                day = gr.Slider(1, 28, value=default[4], step=1, label="Pickup day")
                distance = gr.Number(value=default[5], label="Distance (miles)")
                market_index = gr.Number(value=default[6], label="Market index")
                run = gr.Button("Quote rate", variant="primary")
            with gr.Column():
                rate_out = gr.Textbox(label="Predicted rate ($)", interactive=False)
                rpm_out = gr.Textbox(label="Predicted $/mile", interactive=False)
                detail = gr.Markdown(metrics_markdown())
                chart = gr.Image(
                    value=str(DECEMBER_CHART) if DECEMBER_CHART.is_file() else None,
                    label="December forecast (Lexington → Fort Wayne)",
                )

        preset.change(fn=apply_preset, inputs=[preset], outputs=[lane, equipment, weight, month, day, distance, market_index])
        run.click(
            fn=quote_lane,
            inputs=[lane, equipment, weight, month, day, distance, market_index],
            outputs=[rate_out, rpm_out, detail, chart],
        )
        demo.load(
            fn=lambda: quote_lane(*default),
            outputs=[rate_out, rpm_out, detail, chart],
        )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
