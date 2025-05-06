# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Prepare your data (example with dummy data)
# X = features, y = target
X = np.array([[1], [2], [3], [4], [5]])  # Example feature (single feature for simplicity)
y = np.array([2, 4, 5, 4, 5])             # Example target

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the linear regression model
model = LinearRegression()

# 4. Train the model (fit to the training data)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination (R²): %.2f" % r2_score(y_test, y_pred))

# 7. Use the model for new predictions
new_data = np.array([[6]])
prediction = model.predict(new_data)
print("Prediction for input 6:", prediction)