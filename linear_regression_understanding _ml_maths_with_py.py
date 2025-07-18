import numpy as np
import matplotlib.pyplot as plt
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        fw = x[i]*w + b
        cost += (fw-y[i])**2
    cost /= 2*m
    return cost

def gradient_descent(x,y,w,b):  
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        fwb = x[i] * w + b
        dj_dw += (fwb - y[i]) * x[i]
        dj_db += (fwb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def findwb(x, y, alpha=0.01, num_iters=1000):
    m = x.shape[0]
    w = 0.0
    b = 0.0
    J_history = []
    acc_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_descent(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        y_pred = w * x + b
        acc = r2_score(y, y_pred)
        acc_history.append(acc)
        
        J_history.append(compute_cost(x, y, w, b))
        if i % 100 == 0:
            print(f"Iteration {i}: cost = {J_history[-1]}, w = {w}, b = {b}, acc = {acc_history[-1]}")
    print(f"final cost: {J_history[-1]}, w = {w}, b = {b}, acc = {acc_history[-1]}")
        
    return w, b, J_history


x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([5.1, 7.0, 9.2, 11.1, 13.0])


w, b, j_history = findwb(x_train,y_train,alpha=0.1, num_iters=1000)
y_pred = w * x_train + b

r2 = r2_score(y_train, y_pred)

print(f"\nModel Evaluation:")
print(f"R² Score = {r2}")

plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.plot(x_train, y_pred, color='red', label='Prediction')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
