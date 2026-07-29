"""Prediction engine for single-machine scoring."""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional
import numpy as np
import joblib
import pandas as pd

from src.preprocess import FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

RISK_THRESHOLDS = {
    "LOW": (0.0, 0.25),
    "MEDIUM": (0.25, 0.55),
    "HIGH": (0.55, 0.80),
    "CRITICAL": (0.80, 1.01),
}

ACTIONS = {
    "LOW": "✅ Continue normal operation. Schedule next routine inspection per standard cycle.",
    "MEDIUM": "⚠️ Monitor closely. Inspect tool wear and torque sensors within 48 hours.",
    "HIGH": "🔴 Prioritize maintenance. Reduce load and inspect within 8 hours.",
    "CRITICAL": "🚨 Immediate shutdown recommended. Failure imminent — urgent inspection required.",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}


@dataclass
class MachineInput:
    machine_type: str          # "L", "M", or "H"
    air_temperature: float     # Kelvin
    process_temperature: float # Kelvin
    rotational_speed: float    # RPM
    torque: float              # Nm
    tool_wear: float           # minutes


def _get_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        from src.train import train
        train()
    return joblib.load(model_path)


def predict_failure(inp: MachineInput) -> dict:
    model = _get_model()
    type_encoded = TYPE_MAP.get(inp.machine_type.upper(), 0)
    row = pd.DataFrame([[
        type_encoded,
        inp.air_temperature,
        inp.process_temperature,
        inp.rotational_speed,
        inp.torque,
        inp.tool_wear,
    ]], columns=FEATURES)

    prob = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])

    risk = "LOW"
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            risk = level
            break

    return {
        "failure_prediction": pred,
        "failure_probability": round(prob, 4),
        "risk_category": risk,
        "recommended_action": ACTIONS[risk],
    }


if __name__ == "__main__":
    result = predict_failure(MachineInput("L", 300.0, 310.0, 1500, 40.0, 120))
    print(result)
