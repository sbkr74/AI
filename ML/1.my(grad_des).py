import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 200
X = np.linspace(0, 10, n_samples)

# True parameters
true_slope = 2.5
true_intercept = 1.0

# Generate y with noise
noise = np.random.normal(0, 1.5, n_samples)
y_true = true_slope * X + true_intercept
y = y_true + noise

# Add outliers
outlier_indices = np.random.choice(n_samples, size=15, replace=False)
y[outlier_indices] += np.random.normal(15, 5, size=15)  # Large noise for outliers

# Reshape X for sklearn
X_reshaped = X.reshape(-1,1)

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X_reshaped,y)

# Predictions
y_pred = lr.predict(X_reshaped)

# Calculate MSE
mse = mean_squared_error(y,y_pred)
print(f"OLS Coefficients: slope={lr.coef_[0]:.2f}, intercept={lr.intercept_:.2f}")
print(f"Mean Squared Error: {mse:.2f}")


class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # Parameters (intercept, slope)
        self.loss_history = []
    
    def fit(self, X, y):
        # Add bias term (column of 1s)
        X_b = np.c_[np.ones((len(X), 1)), X]
        
        # Initialize parameters
        self.theta = np.random.randn(2, 1)
        
        for iteration in range(self.n_iterations):
            gradients = 2/len(X) * X_b.T.dot(X_b.dot(self.theta) - y.reshape(-1, 1))
            self.theta -= self.learning_rate * gradients
            
            # Calculate and store loss (MSE)
            mse = np.mean((X_b.dot(self.theta) - y.reshape(-1, 1))**2)
            self.loss_history.append(mse)
    
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)
    
    def get_params(self):
        return {'intercept': self.theta[0][0], 'slope': self.theta[1][0]}

# Initialize and fit gradient descent model
gd_lr = GradientDescentLinearRegression(learning_rate=0.01, n_iterations=1000)
gd_lr.fit(X_reshaped, y)

# Get predictions
y_pred_gd = gd_lr.predict(X_reshaped)

# Get parameters
params = gd_lr.get_params()
print(f"Gradient Descent Coefficients: slope={params['slope']:.2f}, intercept={params['intercept']:.2f}")

# Calculate MSE
mse_gd = mean_squared_error(y, y_pred_gd)
print(f"Mean Squared Error (Gradient Descent): {mse_gd:.2f}")


# Plot results comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data', alpha=0.7)
plt.plot(X, y_true, color='red', label='True relationship', linewidth=2)
plt.plot(X, y_pred, color='green', label='OLS prediction', linewidth=2)
plt.plot(X, y_pred_gd, color='blue', linestyle='--', label='Gradient Descent prediction', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Regression Methods')
plt.legend()
plt.show()

# Feature scaling (normalization)
X_normalized = (X_reshaped - np.mean(X_reshaped)) / np.std(X_reshaped)

# Initialize and fit gradient descent model with normalized features
gd_lr_improved = GradientDescentLinearRegression(learning_rate=0.1, n_iterations=2000)
gd_lr_improved.fit(X_normalized, y)

# Get predictions (need to normalize test data the same way)
y_pred_gd_improved = gd_lr_improved.predict(X_normalized)

# Get parameters (need to adjust for scaling)
mean_X = np.mean(X_reshaped)
std_X = np.std(X_reshaped)
params_improved = gd_lr_improved.get_params()
adjusted_intercept = params_improved['intercept'] - params_improved['slope'] * mean_X / std_X
adjusted_slope = params_improved['slope'] / std_X

print(f"Improved Gradient Descent Coefficients: slope={adjusted_slope:.2f}, intercept={adjusted_intercept:.2f}")

# Calculate MSE
mse_gd_improved = mean_squared_error(y, y_pred_gd_improved)
print(f"Mean Squared Error (Improved Gradient Descent): {mse_gd_improved:.2f}")

# Plot results comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data', alpha=0.7)
plt.plot(X, y_true, color='red', label='True relationship', linewidth=2)
plt.plot(X, y_pred, color='green', label='OLS prediction', linewidth=2)
plt.plot(X, y_pred_gd, color='blue', linestyle='--', label='Basic GD prediction', linewidth=2)
plt.plot(X, y_pred_gd_improved, color='purple', linestyle='-.', label='Improved GD prediction', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Gradient Descent Implementations')
plt.legend()
plt.show()

class RidgeGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha  # Regularization strength
        self.theta = None
        self.loss_history = []
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(2, 1)
        
        for iteration in range(self.n_iterations):
            gradients = 2/len(X) * X_b.T.dot(X_b.dot(self.theta) - y.reshape(-1, 1))
            # Add regularization term (excluding intercept)
            gradients[1:] += 2 * self.alpha * self.theta[1:] / len(X)
            self.theta -= self.learning_rate * gradients
            
            # Calculate loss (MSE + regularization term)
            mse = np.mean((X_b.dot(self.theta) - y.reshape(-1, 1))**2)
            reg_term = self.alpha * np.sum(self.theta[1:]**2)
            self.loss_history.append(mse + reg_term)
    
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)
    
    def get_params(self):
        return {'intercept': self.theta[0][0], 'slope': self.theta[1][0]}

# Initialize and fit ridge regression with gradient descent
ridge_gd = RidgeGradientDescent(learning_rate=0.1, n_iterations=2000, alpha=5.0)
ridge_gd.fit(X_normalized, y)

# Get predictions
y_pred_ridge = ridge_gd.predict(X_normalized)

# Get parameters (adjusted for scaling)
params_ridge = ridge_gd.get_params()
ridge_intercept = params_ridge['intercept'] - params_ridge['slope'] * mean_X / std_X
ridge_slope = params_ridge['slope'] / std_X

print(f"Ridge Regression Coefficients: slope={ridge_slope:.2f}, intercept={ridge_intercept:.2f}")

# Calculate MSE
mse_ridge = mean_squared_error(y, y_pred_ridge)
print(f"Mean Squared Error (Ridge Regression): {mse_ridge:.2f}")

# Plot all results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data', alpha=0.7)
plt.plot(X, y_true, color='red', label='True relationship', linewidth=2)
plt.plot(X, y_pred, color='green', label='OLS', linewidth=2)
plt.plot(X, y_pred_gd_improved, color='blue', linestyle='--', label='Improved GD', linewidth=2)
plt.plot(X, y_pred_ridge, color='purple', linestyle='-.', label='Ridge Regression', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of All Regression Methods')
plt.legend()
plt.show()