import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

X_train = np.array([[952, 2, 1, 65], [1244, 3, 2, 64], [1947, 3, 2, 17]])
y_train = np.array([271.5, 232, 509.8])
X_features = ['size(sqft)', 'no of bedrooms', 'floors', 'age of house']


def zscoreNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_normalized = (X - mu) / sigma

    return X_normalized, mu, sigma

X_normalized, mu, sigma = zscoreNormalize(X_train)

print(X_normalized)

#Now use gradient descent on normalized dataset to find W and B and plot predicted points using original training set and new W and B



