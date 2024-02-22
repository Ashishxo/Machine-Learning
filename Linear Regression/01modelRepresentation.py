import numpy as np
import matplotlib.pyplot as plt

def prediction(x, w, b):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb
        


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = len(x_train)
w= 200
b = 100

f_wb = prediction(x_train, w, b)

plt.plot(x_train, f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, c='r', marker='x', label="Real Values")
plt.title("House Prices")
plt.show()


print(f"The cost of house with 1200 sqft is abour {w*1.2+b}")