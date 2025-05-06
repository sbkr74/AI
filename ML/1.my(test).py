import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 200
X = np.linspace(0, 10, n_samples)

noise = np.random.normal(0, 1.5, n_samples)
y = X*2.3+noise


# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the Linear Regression model
model = LinearRegression()

# 4. Train the model (fit to the training data)
X_train = X_train.reshape(-1,1)
model.fit(X_train,y_train)

# 5. Make predictions
X_test = X_test.reshape(-1,1)
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination (R²): %.2f" % r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data', alpha=0.7)
plt.plot(X, y, color='red', label='True relationship', linewidth=2)
plt.plot(X_test, y_pred, color='green', label='OLS prediction', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ordinary Least Squares Regression')
plt.legend()
plt.show()