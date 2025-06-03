import numpy as np

def generate_data(n_samples=100, noise=0.1):
    """
    Generate synthetic linear data for regression.
    
    Parameters:
    n_samples (int): Number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the output.
    
    Returns:
    X (np.ndarray): Input features of shape (n_samples, 1).
    Y (np.ndarray): Output targets of shape (n_samples,).
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(n_samples, 1) * 10  # Random values between 0 and 10
    true_slope = 2.0
    true_intercept = 3.0
    Y = true_slope * X.flatten() + true_intercept + np.random.normal(0, noise, n_samples)
    
    return X, Y
    
def train_test_split(X, y, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    X (np.ndarray): Input features.
    y (np.ndarray): Output targets.
    test_size (float): Proportion of the dataset to include in the test split.
    
    Returns:
    X_train (np.ndarray): Training input features.
    X_test (np.ndarray): Testing input features.
    y_train (np.ndarray): Training output targets.
    y_test (np.ndarray): Testing output targets.
    """
    n_samples = X.shape[0]
    split_index = int(n_samples * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def loss_function(y_pred, y_true):
    """
    Compute the mean squared error loss.
    
    Parameters:
    y_pred (np.ndarray): Predicted output values.
    y_true (np.ndarray): True output values.
    
    Returns:
    float: Mean squared error between predicted and true values.
    """
    return np.mean((y_pred - y_true) ** 2)

def forward_pass(X, W, b):
    """
    Forward pass to compute predictions.
    
    Parameters:
    X (np.ndarray): Input features.
    W (float): Weight of the linear model.
    b (float): Bias of the linear model.
    
    Returns:
    np.ndarray: Predicted output values.
    """
    return W * X.flatten() + b