from least_squares import generate_data, train_test_split, forward_pass, loss_function

# Constants for the linear model
W = 0.5 # Weight
b = 2.0 # Bias

# Generate synthetic data and split it into training and testing sets
X, Y = generate_data(n_samples=100, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Forward pass to compute predictions
y_pred = forward_pass(X_train, W, b)

# Calculate the error between predicted and true values
error = loss_function(y_pred, y_train)

# Print the error
print(f"Mean Squared Error: {error:.4f}")