import numpy as np

from app.data import load_diabetes_dataset
from app.model import build_ridge_model, train_model
from app.preprocess import split_data


def test_model_trains_and_predicts():
    X, y = load_diabetes_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)

    model = build_ridge_model(alpha=1.0)
    model = train_model(model, X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape[0] == y_test.shape[0]
    assert np.isfinite(preds).all()
