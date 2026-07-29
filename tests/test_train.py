import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import generate_synthetic_data, preprocess, split_data
from src.train import MODELS

def test_all_models_train():
    df = generate_synthetic_data(500)
    X, y, le, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})
