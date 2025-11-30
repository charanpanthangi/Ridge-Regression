"""Model definition for Ridge Regression."""
from typing import Tuple

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(alpha: float = 1.0) -> Pipeline:
    """
    Build a Ridge Regression model wrapped in a Pipeline with scaling.

    Parameters
    ----------
    alpha : float, optional
        Regularization strength. Higher values mean more shrinkage of coefficients.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline that scales features then fits Ridge.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def train_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Fit the provided Ridge model on the training data.

    Parameters
    ----------
    model : Pipeline
        The Ridge pipeline to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.

    Returns
    -------
    Pipeline
        The fitted model.
    """
    return model.fit(X_train, y_train)


def predict(model: Pipeline, X_test: pd.DataFrame) -> pd.Series:
    """
    Generate predictions from a fitted Ridge model.

    Parameters
    ----------
    model : Pipeline
        Fitted Ridge pipeline.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    pd.Series
        Predicted target values.
    """
    preds = model.predict(X_test)
    return pd.Series(preds, index=X_test.index, name="prediction")


__all__ = ["build_ridge_model", "train_model", "predict"]
