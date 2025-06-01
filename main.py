import least_squares

X, y = least_squares.generate_data(n_samples=100, noise=0.1)
X_train, X_test, y_train, y_test = least_squares.train_test_split(X, y, test_size=0.2)