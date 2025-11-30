# Ridge Regression Tutorial Template

This repository is a beginner-friendly mini-tutorial and template for training a Ridge Regression model (linear regression with L2 regularization) on the scikit-learn **diabetes** dataset. It demonstrates a clean workflow that separates data loading, preprocessing, modeling, evaluation, and visualization so you can learn the concepts and reuse the code in your own projects.

## Why Ridge Regression?

* **Regularization**: Ridge adds an L2 penalty to discourage large coefficients, helping reduce overfitting.
* **Stability**: Works well when features are correlated or when you have more features than observations.
* **Alpha controls strength**: A higher `alpha` increases the penalty; a lower `alpha` behaves closer to standard linear regression.
* **When it shines**: Ridge often outperforms ordinary least squares when multicollinearity exists or the dataset is small/noisy.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── data.py           # Load the diabetes dataset
│   ├── preprocess.py     # Train/test split and scaling
│   ├── model.py          # Ridge model (optionally in a pipeline)
│   ├── evaluate.py       # Regression metrics
│   ├── visualize.py      # Optional prediction plot
│   ├── main.py           # End-to-end script
├── notebooks/
│   └── demo_ridge_regression.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── examples/
│   └── README_examples.md
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

## Quickstart

### 1) Set up the environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Run the end-to-end script
```bash
python app/main.py
```
This will:
1. Load the diabetes dataset.
2. Split into train/test sets and scale the features (important because Ridge is sensitive to feature scale).
3. Train a Ridge Regression model.
4. Evaluate with MSE, MAE, RMSE, and R².
5. Save a predicted-vs-actual plot to `examples/pred_vs_actual.svg`.

### 3) Explore the notebook
Launch Jupyter and open the tutorial notebook:
```bash
jupyter notebook notebooks/demo_ridge_regression.ipynb
```
The notebook walks through the intuition behind Ridge, the role of `alpha`, training, evaluation, and visualization.

### 4) Run tests
```bash
pytest
```

### 5) Run with Docker
```bash
docker build -t ridge-regression .
docker run --rm ridge-regression
```

## Dataset

The [scikit-learn diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) is a classic regression benchmark with 10 features and a continuous target representing a quantitative measure of disease progression one year after baseline.

## Implementation Notes

* Scaling with `StandardScaler` is crucial for Ridge because the L2 penalty depends on coefficient magnitudes; without scaling, features on larger scales would dominate.
* The default `alpha=1.0` provides a modest amount of regularization. Try smaller values (closer to OLS) or larger ones (more shrinkage) to see how metrics change.
* The code uses a scikit-learn `Pipeline` so scaling and modeling are bound together safely during training and inference.

## Future Extensions

* Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
* Alpha comparison plots to visualize how regularization strength affects performance.
* Cross-validation for more robust evaluation.
* Logging and experiment tracking (e.g., MLflow or Weights & Biases).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
