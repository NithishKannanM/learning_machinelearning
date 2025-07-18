import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Compute cross-entropy cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    z = np.dot(x, w) + b
    fwb = sigmoid(z)
    cost = -np.sum(y * np.log(fwb + 1e-15) + (1 - y) * np.log(1 - fwb + 1e-15)) / m
    return cost

# Compute gradients
def gradient_descent(x, y, w, b):  
    m = x.shape[0]
    z = np.dot(x, w) + b
    fwb = sigmoid(z)
    error = fwb - y
    dj_dw = np.dot(x.T, error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

# Train the model using gradient descent
def findwb(x, y, w=None, b=0, alpha=0.01, num_iters=1000):
    m, n = x.shape
    J_history = []

    if w is None:
        w = np.zeros(n)

    for i in range(num_iters):
        dj_dw, dj_db = gradient_descent(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))
        
        if i % 100 == 0:
            print(f"Iter {i}: Cost = {J_history[-1]:.4f}, w = {w}, b = {b:.4f}")

    print(f"\nFinal: Cost = {J_history[-1]:.4f}, w = {w}, b = {b:.4f}")
    return w, b, J_history

# Predict probabilities and binary output
def predict(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int), probs

# Simple dataset: 2 features, binary labels
x = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [5, 6],
              [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train
w, b, cost_history = findwb(x, y, alpha=0.1, num_iters=1000)

# Predict
x_test = np.array([[2, 2], [6, 6]])
preds, probs = predict(x_test, w, b)

# Output
print("\nPredictions:", preds)
print("Probabilities:", probs)


# 1. Plot cost history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost History")

# 2. Plot data points and decision boundary
plt.subplot(1, 2, 2)
# Plot data points
plt.scatter(x[y==0][:,0], x[y==0][:,1], color='red', label='Class 0')
plt.scatter(x[y==1][:,0], x[y==1][:,1], color='blue', label='Class 1')

# Plot test points
plt.scatter(x_test[:,0], x_test[:,1], color='green', marker='x', s=100, label='Test Points')

# Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
x1_vals = np.linspace(x[:,0].min()-1, x[:,0].max()+1, 100)
if w[1] != 0:
    x2_vals = -(w[0]*x1_vals + b)/w[1]
    plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data & Decision Boundary")
plt.legend()
plt.tight_layout()
plt.show()
