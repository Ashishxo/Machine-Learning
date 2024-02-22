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
    d_djw = 0
    d_djb = 0

    for i in range(m):
        d_djw += ((w*x[i]+b) - y[i]) * x[i]
        d_djb += (w*x[i]+b) - y[i]

    d_djb /= m
    d_djw /= m

    return d_djw, d_djb


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    i_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            i_history.append(i)
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i}: Cost {J_history[i]} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}")
 
    return w, b, J_history, i_history


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w = 100
b = 100

w_final, b_final, jHist, iHist = gradient_descent(x_train, y_train, w, b, 0.01, 10000, compute_cost, gradient)


plt.plot(iHist, jHist, c='r')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print(w_final)
print(b_final)



