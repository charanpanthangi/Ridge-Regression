"""Data loading utilities for the Ridge Regression tutorial."""
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes


def load_diabetes_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the diabetes dataset from scikit-learn.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` as a DataFrame and target vector ``y`` as a Series.
    """
    dataset = load_diabetes()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


__all__ = ["load_diabetes_dataset"]
