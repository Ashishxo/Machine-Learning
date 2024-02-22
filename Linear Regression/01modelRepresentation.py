import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = len(x_train)
w= 200
b = 100


# print(f"The number of training examples is: {m}")


# i =  1

# x_i = x_train[i]
# y_i = y_train[i]
# print(f"(x^({i+1}), y^({i+1})) = ({x_i}, {y_i})")

# plt.scatter(x_train, y_train, marker='x', c='r')
# plt.title("House Prices Prediction")

# plt.ylabel("Price in 1000's of Dollars")
# plt.xlabel("Size (1000 sqft)")
# plt.show()

f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, f_wb, c='b', label="Prediction")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.ylabel("House Prices")
plt.xlabel("Size (1000 sqft)")
plt.show()


print(f"The cost of house with 1200 sqft is abour {w*1.2+b}")