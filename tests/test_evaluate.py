import numpy as np
import pandas as pd

from app.evaluate import regression_metrics


def test_regression_metrics_outputs_numeric():
    y_true = pd.Series([3.0, -0.5, 2.0, 7.0])
    y_pred = pd.Series([2.5, 0.0, 2.0, 8.0])

    metrics = regression_metrics(y_true, y_pred)

    for value in metrics.values():
        assert isinstance(value, float)
        assert np.isfinite(value)

    assert metrics["mse"] >= 0
    assert -1 <= metrics["r2"] <= 1
