import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])     

w = 200
b = 100


def compute_cost(x, y, w, b): 
    m = x.shape[0]

    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost_sum += (f_wb - y[i]) ** 2
    
    total_cost = 1/(2*m) * cost_sum
    return total_cost
        


print(compute_cost(x_train, y_train, w, b))