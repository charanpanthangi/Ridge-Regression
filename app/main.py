"""Command-line entry point for training and evaluating Ridge Regression."""
from pathlib import Path

from app.data import load_diabetes_dataset
from app.evaluate import format_metrics, regression_metrics
from app.model import build_ridge_model, predict, train_model
from app.preprocess import split_data
from app.visualize import DEFAULT_SAVE_PATH, plot_predictions


def run(alpha: float = 1.0, test_size: float = 0.2, random_state: int = 42) -> None:
    """Run the full Ridge Regression workflow with the specified parameters."""
    X, y = load_diabetes_dataset()

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_ridge_model(alpha=alpha)
    model = train_model(model, X_train, y_train)

    predictions = predict(model, X_test)
    metrics = regression_metrics(y_test, predictions)

    print(format_metrics(metrics))

    save_path: Path = plot_predictions(
        y_true=y_test, y_pred=predictions, save_path=DEFAULT_SAVE_PATH
    )
    print(f"Saved prediction plot to {save_path}")


def main() -> None:
    """Default CLI entry point with alpha=1.0."""
    run(alpha=1.0)


if __name__ == "__main__":
    main()
