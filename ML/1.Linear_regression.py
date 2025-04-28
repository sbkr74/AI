import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
# X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature
# y = np.array([2, 4, 5, 4, 5])                  # Target
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1.2, 1.8, 2.6, 3.2, 3.8])
# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Model evaluation
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mean_squared_error(y, y_pred)}")
print(f"R²: {r2_score(y, y_pred)}")

# Plotting
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()