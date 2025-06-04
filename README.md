# Least Squares Linear Regression from Scratch

This project implements simple linear regression using the least squares method, built entirely from scratch with NumPy. No machine learning libraries are used. The code generates synthetic data, splits it into training and testing sets, fits a linear model, and evaluates its performance.

## Features

- Synthetic data generation with adjustable noise
- Manual train/test split
- Closed-form solution for linear regression
- Mean Squared Error (MSE) calculation for evaluation
- Minimal dependencies (NumPy only)

## Project Structure

```
.
├── least_squares.py      # Core implementation (data, split, error)
├── main.py               # Entry point (to be implemented)
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
├── .gitignore
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/tamimmirza/Least-Squares-Linear-Regression.git
    cd Least-Squares Linear Regression
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Implement your workflow in `main.py` (e.g., generate data, split, fit, evaluate).
2. Run the script:
    ```sh
    python main.py
    ```

## Example

```python
from least_squares import create_synthetic_data, train_test_split, error_function

# Generate data
X, Y = create_synthetic_data(n_samples=100, noise=0.2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit model (implement fitting in main.py)
# ...

# Predict and evaluate (implement prediction in main.py)
# mse = error_function(y_pred, y_test)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

TAMIM-UL-HAQ MIRZA  
[Email](mailto:itstamimmirza@gmail.com)
