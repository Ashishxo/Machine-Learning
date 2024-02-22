import math, copy
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = (1 / (2*m)) * cost

    return total_cost


def gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        dj_dw += ((w * x[i] + b) - y[i]) * x[i]
        dj_db += (w * x[i] + b) - y[i]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha, iterations, gradient_function):

    for i in range(iterations):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b



x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w = 100
b = 100

w_final, b_final = gradient_descent(x_train, y_train, w, b, 0.01, 10000, gradient)

print(w_final)
print(b_final)



