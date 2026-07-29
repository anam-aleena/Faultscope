"""
PredictX - Data Pipeline Module
Handles ingestion, validation, and preprocessing of AI4I 2020 dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

DATA_RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

FEATURE_COLS = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
TARGET_COL = 'Machine failure'


def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic AI4I-like dataset when UCI is unavailable."""
    rng = np.random.RandomState(seed)
    types = rng.choice(['L', 'M', 'H'], size=n_samples, p=[0.6, 0.3, 0.1])
    air_temp = rng.normal(300, 2, n_samples)
    proc_temp = air_temp + rng.normal(10, 1, n_samples)
    rot_speed = rng.normal(1538, 179, n_samples)
    torque = rng.normal(40, 10, n_samples)
    tool_wear = rng.uniform(0, 253, n_samples)

    # Failure logic: high torque + high tool wear increases risk
    failure_prob = (
        0.02
        + 0.05 * (torque > 60).astype(float)
        + 0.08 * (tool_wear > 200).astype(float)
        + 0.03 * (rot_speed < 1200).astype(float)
        + 0.04 * (types == 'L').astype(float)
    )
    failure_prob = np.clip(failure_prob, 0, 1)
    failure = rng.binomial(1, failure_prob, n_samples)

    df = pd.DataFrame({
        'UDI': range(1, n_samples + 1),
        'Product ID': [f'{t}{rng.randint(10000,99999)}' for t in types],
        'Type': types,
        'Air temperature [K]': air_temp.round(1),
        'Process temperature [K]': proc_temp.round(1),
        'Rotational speed [rpm]': rot_speed.round(0).astype(int),
        'Torque [Nm]': torque.round(1),
        'Tool wear [min]': tool_wear.round(0).astype(int),
        'Machine failure': failure,
        'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
    })
    return df


def load_data() -> pd.DataFrame:
    """Load dataset — tries UCI CSV first, falls back to synthetic."""
    csv_path = os.path.join(DATA_RAW, 'ai4i2020.csv')
    if os.path.exists(csv_path):
        print("[DataPipeline] Loading from local CSV...")
        df = pd.read_csv(csv_path)
    else:
        print("[DataPipeline] Generating synthetic AI4I-like dataset...")
        df = generate_synthetic_data()
        os.makedirs(DATA_RAW, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Run quality checks and return a report dict."""
    report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'failure_rate': float(df[TARGET_COL].mean()),
        'class_distribution': df[TARGET_COL].value_counts().to_dict(),
    }
    return report


def preprocess(df: pd.DataFrame):
    """Encode, split and return X_train, X_test, y_train, y_test + encoder."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    le = LabelEncoder()
    df['Type_enc'] = le.fit_transform(df['Type'])

    feature_cols_enc = [
        'Type_enc', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    X = df[feature_cols_enc]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, le


def run_pipeline():
    """Full pipeline: load → validate → preprocess → save."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    df = load_data()
    report = validate_data(df)

    report_path = os.path.join(DATA_PROCESSED, 'quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[DataPipeline] Quality report saved → {report_path}")
    print(f"  Rows: {report['total_rows']} | Failure rate: {report['failure_rate']:.2%}")
    print(f"  Duplicates: {report['duplicate_rows']} | Missing: {sum(report['missing_values'].values())}")

    df.to_csv(os.path.join(DATA_PROCESSED, 'cleaned_data.csv'), index=False)
    return df, report


if __name__ == '__main__':
    run_pipeline()
