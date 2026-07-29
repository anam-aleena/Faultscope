import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import generate_synthetic_data, preprocess, split_data, FEATURES, TARGET

def test_synthetic_data_shape():
    df = generate_synthetic_data(500)
    assert len(df) == 500
    assert TARGET in df.columns

def test_preprocess_output():
    df = generate_synthetic_data(500)
    X, y, le, schema = preprocess(df)
    assert X.shape[1] == len(FEATURES)
    assert len(X) == len(y)

def test_split_sizes():
    df = generate_synthetic_data(1000)
    X, y, le, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
