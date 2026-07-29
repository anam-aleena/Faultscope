import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _ensure_model():
    from src.preprocess import generate_synthetic_data, preprocess, split_data
    from src.train import MODELS, MODELS_DIR
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        df = generate_synthetic_data(800)
        X, y, le, _ = preprocess(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = MODELS["Random Forest"]
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

from src.predict import MachineInput, predict_failure

def test_predict_returns_keys():
    _ensure_model()
    result = predict_failure(MachineInput("L", 300.0, 310.0, 1500, 40.0, 120))
    assert "failure_prediction" in result
    assert "failure_probability" in result
    assert "risk_category" in result

def test_probability_in_range():
    _ensure_model()
    result = predict_failure(MachineInput("H", 303.0, 313.0, 1200, 65.0, 230))
    assert 0.0 <= result["failure_probability"] <= 1.0

def test_risk_category_valid():
    _ensure_model()
    result = predict_failure(MachineInput("M", 300.0, 310.0, 1538, 40.0, 100))
    assert result["risk_category"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
