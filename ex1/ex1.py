import numpy as np
import matplotlib.pyplot as plt


def test():
    data = np.loadtxt("ex1data1.txt", delimiter=',', dtype={'names': ('population', 'profit'), 'formats': (float, float)})
    X1 = data['population']
    y = data['profit']
    m = len(y)
    print(X1)
    warmUpExercise()


def warmUpExercise():
    return np.eye(5)


def computeCost(X, y, theta):
    print(X, y, theta)
    return np.float(0)


def gradientDescent(X, y, theta, alpha, num_iters):
    return np.array([0, 0])


def featureNormalize(X):
    return np.ones(20)


def computeCostMulti(X, y, theta):
    return np.float(0)


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    return np.array([0, 0])


def normalEqn(X, y):
    return np.array([0, 0])