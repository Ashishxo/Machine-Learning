import numpy as np
import matplotlib.pyplot as plt
import copy, math

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost=0.0

    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost += (f_wb_i - y[i]) ** 2

    cost /= 2 * m
    return cost


def gradient(X, y, w, b):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db= 0.

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def compute_gradient(X, y, w_in, b_in, cost_function, gradFunction, alpha, iterations):
    w_final = copy.deepcopy(w_in)
    b_final = b_in
    J_history = []
    I_history = []
    for i in range(iterations):
        dj_dw, dj_db = gradFunction(X, y, w_final, b_final)

        

        w_final = w_final - alpha * dj_dw
        b_final = b_final - alpha * dj_db

        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w_final, b_final))
            I_history.append(i)

    return w_final, b_final, J_history, I_history

    
        


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


b_init = 0
w_init = np.array([ 0, 0, 0, 0])


w, b, jHist, iHist = compute_gradient(X_train, y_train, w_init, b_init, compute_cost, gradient, 5.2e-7, 1000) 

print(w)
print(b)


plt.plot(iHist, jHist)
plt.show()