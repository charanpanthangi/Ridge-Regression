"""Evaluation utilities for regression models."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute common regression metrics.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target values.
    y_pred : pd.Series
        Predicted target values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing MSE, MAE, RMSE, and R^2 scores.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Nicely format regression metrics for printing.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dictionary.

    Returns
    -------
    str
        Human-readable summary of metrics.
    """
    lines = [
        "Model performance:",
        f"  MSE : {metrics['mse']:.4f}",
        f"  RMSE: {metrics['rmse']:.4f}",
        f"  MAE : {metrics['mae']:.4f}",
        f"  R^2 : {metrics['r2']:.4f}",
    ]
    return "\n".join(lines)


__all__ = ["regression_metrics", "format_metrics"]
