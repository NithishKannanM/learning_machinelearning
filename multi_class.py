import numpy as np
import matplotlib.pyplot as plt
def compute_cost(x, y, w, b):
    m,n = x.shape
    fw = np.dot(x,w) + b
    cost = np.sum((fw - y) ** 2) / (2 * m)
    return cost

def gradient_descent(x,y,w,b):  
    m, n = x.shape
    fwb = (np.dot(x,w) + b )- y
    dj_dw = np.dot(x.T,fwb)/m
    dj_db = np.sum(fwb)/m
    return dj_dw, dj_db

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def findwb(x, y, alpha=0.01, num_iters=1000):
    m, n = x.shape
    w = np.zeros(n)
    b = 0.0
    J_history = []
    acc_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_descent(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i % 100 == 0:
            acc = r2_score(y, np.dot(x, w) + b)
            acc_history.append(acc)
            J_history.append(compute_cost(x, y, w, b))
            print(f"Iteration {i}: cost = {J_history[-1]}, w = {w}, b = {b}, acc = {acc_history[-1]}")
    print(f"final cost: {J_history[-1]}, w = {w}, b = {b}, acc = {acc_history[-1]}")
        
    return w, b, J_history


X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y_train = np.array([15, 20, 25, 30, 35])


w, b, j_history = findwb(X_train,y_train,alpha=0.01, num_iters=1000)
plt.plot(j_history)
plt.xlabel("Iterations (per 100)")
plt.ylabel("Cost (Loss)")
plt.title("📉 Cost Reduction Over Time")
plt.grid(True)
plt.show()
