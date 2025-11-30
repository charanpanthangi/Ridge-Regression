"""Visualization utilities for Ridge Regression results."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SAVE_PATH = Path("examples/pred_vs_actual.svg")


def plot_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    save_path: Optional[Path] = None,
    title: str = "Predicted vs Actual",
) -> Path:
    """
    Plot predicted vs. actual target values and save as an SVG.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target values.
    y_pred : pd.Series
        Predicted target values.
    save_path : Optional[Path], optional
        Path to save the figure. Defaults to ``examples/pred_vs_actual.svg``.
    title : str, optional
        Plot title.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, edgecolor="white")
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.close()
    return save_path


__all__ = ["plot_predictions", "DEFAULT_SAVE_PATH"]
