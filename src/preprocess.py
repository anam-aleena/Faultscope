"""Data ingestion, validation, and preprocessing for PredictX."""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"


def _make_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)


def generate_synthetic_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate an AI4I-like synthetic dataset as offline fallback."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    air_temp = rng.normal(300, 2, n)
    proc_temp = air_temp + rng.normal(10, 1, n)
    rpm = rng.normal(1538, 179, n).clip(1168, 2886)
    torque = rng.normal(40, 10, n).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 254, n).astype(float)

    # Failure logic based on real dataset patterns
    failure_prob = (
        0.01
        + 0.04 * (tool_wear > 200).astype(float)
        + 0.03 * (torque > 60).astype(float)
        + 0.02 * (rpm < 1300).astype(float)
        + 0.02 * ((proc_temp - air_temp) > 11.5).astype(float)
    )
    failure = (rng.random(n) < failure_prob).astype(int)

    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"{t}{rng.integers(10000,99999)}" for t in types],
        "Type": types,
        "Air temperature [K]": air_temp.round(1),
        "Process temperature [K]": proc_temp.round(1),
        "Rotational speed [rpm]": rpm.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": tool_wear.astype(int),
        "Machine failure": failure,
        "TWF": (rng.random(n) < 0.01).astype(int),
        "HDF": (rng.random(n) < 0.01).astype(int),
        "PWF": (rng.random(n) < 0.01).astype(int),
        "OSF": (rng.random(n) < 0.01).astype(int),
        "RNF": (rng.random(n) < 0.001).astype(int),
    })


def load_data() -> pd.DataFrame:
    _make_dirs()
    csv_path = os.path.join(RAW_DIR, "ai4i2020.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "failure_rate": float(df[TARGET].mean()),
        "class_distribution": {int(k): int(v) for k, v in df[TARGET].value_counts().items()},
    }
    return report


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder, dict]:
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURES + [TARGET])

    le = LabelEncoder()
    df = df.copy()
    df["Type"] = le.fit_transform(df["Type"])

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Save schema
    schema = {
        "features": FEATURES,
        "target": TARGET,
        "type_classes": list(le.classes_),
        "feature_stats": X.describe().to_dict(),
    }
    schema_path = os.path.join(MODELS_DIR, "prediction_schema.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    quality_path = os.path.join(PROCESSED_DIR, "quality_report.json")
    with open(quality_path, "w") as f:
        json.dump(validate_data(df), f, indent=2)

    return X, y, le, schema


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
