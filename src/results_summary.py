"""Run the full ML pipeline: data → EDA → train → evaluate → explain."""
from __future__ import annotations

from src.eda import run_eda
from src.train import train
from src.evaluate import evaluate
from src.explain import run_explain


def run_all():
    print("=" * 50)
    print("PredictX: Running full ML pipeline")
    print("=" * 50)
    print("\n[1/4] Running EDA...")
    run_eda()
    print("\n[2/4] Training models...")
    metrics = train()
    print("\n[3/4] Evaluating best model...")
    evaluate()
    print("\n[4/4] Generating explainability assets...")
    run_explain()
    print("\n✅ Pipeline complete.")
    print(f"   Best model: {metrics.get('best_model')}")


if __name__ == "__main__":
    run_all()
