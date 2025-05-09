import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import r2_score

# Generate linear data
np.random.seed(42)
x_linear = np.linspace(0, 10, 100)
y_linear = 2 * x_linear + 3 + np.random.normal(0, 2, 100)

# Generate non-linear data
x_nonlinear = np.linspace(0, 10, 100)
y_nonlinear = 0.5 * x_nonlinear**2 + np.random.normal(0, 2, 100)

# Linear Plot
plt.scatter(x_linear, y_linear, color='blue')
plt.title("Linear Relationship: y = 2x + 3 + noise")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Non-Linear Plot
plt.scatter(x_nonlinear, y_nonlinear, color='red')
plt.title("Non-Linear Relationship: y = 0.5x² + noise")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()