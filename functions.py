import random as r
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

"""function"""


def func(x, y):
    return x ** 2 - 6 * x + y ** 2 - 20 * y + 109


"""derivative of a function on x"""


def fprime_x(X):
    x, y = symbols('x y')
    fprime = lambdify(x, diff(x ** 2 - 6 * x + y ** 2 - 20 * y + 109, x))
    return fprime(X)


"""derivative of a function on y"""


def fprime_y(Y):
    x, y = symbols('x y')
    fprime = lambdify(y, diff(x ** 2 - 6 * x + y ** 2 - 20 * y + 109, y))
    return fprime(Y)


"""Randomly finding better start for a steepest descent"""


def optimal_range():
    best_x = r.randrange(1, 100)
    best_y = r.randrange(1, 100)
    for i in range(1, 100):
        x = r.randrange(1, 100)
        y = r.randrange(1, 100)
        if best_x > fprime_x(x):
            best_x = fprime_x(x)
        if best_y > fprime_y(y):
            best_y = fprime_y(y)
    return best_x, best_y





"""creating a 3d graph function"""


def graph(min_x, min_y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-30, 30, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.scatter(min_x, min_y, func(min_x, min_y), s=50, color='red')

    plt.show()
