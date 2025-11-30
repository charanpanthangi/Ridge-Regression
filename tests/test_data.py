import pandas as pd

from app.data import load_diabetes_dataset


def test_load_diabetes_dataset_shapes():
    X, y = load_diabetes_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    # Diabetes dataset has 10 features
    assert X.shape[1] == 10
