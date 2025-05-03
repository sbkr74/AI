import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise


# Add bias term (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

# Gradient descent parameters
eta = 0.1  # learning rate
n_iterations = 1000
m = 100  # number of samples

# Initialize random weights
theta = np.random.randn(2, 1)

# Store cost history for plotting
cost_history = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    cost = 1/m * np.sum((X_b.dot(theta) - y)**2)
    cost_history.append(cost)

# Results
print(f"Optimal parameters: {theta.ravel()}")
print(f"True parameters would be [4, 3]")

# Plot convergence
# plt.scatter(X,y,color="black")
plt.plot(range(n_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence of Gradient Descent')
plt.show()