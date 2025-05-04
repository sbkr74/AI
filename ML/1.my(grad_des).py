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

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data with noise and outliers', alpha=0.7)
plt.plot(X, y_true, color='red', label='True relationship', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Dataset with Noise and Outliers')
plt.legend()
plt.show()

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

# Plot results
plt.figure(figsize=(10,6))
plt.scatter(X,y,label='Data',alpha=0.7)
plt.plot(X,y_true,color="red",label="True relationship",linewidth=2)
plt.plot(X,y_pred,color="green",label="OLS prediction",linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ordinary Least Square Regression')
plt.legend()
plt.show()
