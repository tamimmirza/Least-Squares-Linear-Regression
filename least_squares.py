import numpy as np

def generate_data(n_samples=100, noise=0.1):
    """
    Create synthetic data for linear regression.
    
    Parameters:
    n_samples (int): Number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the output.
    
    Returns:
    X (np.ndarray): Input features of shape (n_samples, 1).
    y (np.ndarray): Output targets of shape (n_samples,).
    """
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 10  # Random values between 0 and 10
    y = 2 * X.squeeze() + 1 + np.random.normal(0, noise, n_samples)  # Linear relation with noise
    return X, y
    
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

def error_function(Y_pred, Y_true):
    """
    Calculate the mean squared error between predicted and true values.
    
    Parameters:
    Y_pred (np.ndarray): Predicted output values.
    Y_true (np.ndarray): True output values.
    
    Returns:
    float: Mean squared error.
    """
    return np.mean((Y_pred - Y_true) ** 2)

def forward(X, W, b):
    """
    Forward pass to compute predictions.
    
    Parameters:
    X (np.ndarray): Input features.
    W (np.ndarray): Weights of the model.
    b (float): Bias term.
    
    Returns:
    np.ndarray: Predicted output values.
    """
    return X @ W + b