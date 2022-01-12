import functions as f
import numpy as np

'''
    Steepest descent. 
    
    *All functions are in functions.py
    *To change a function - to change the function - override it in the method func() 
        and at prime methods inside the fprime variable   
    *Creating a graph depending on your minimum is not optimized so change graph boundaries 
        manually at graph() function at x,y variables.
'''


## assume everything is a vector
x = np.array(f.optimal_range())
y = np.array(f.optimal_range())

c = 0.8  ## how much imperfection in function improvement we'll settle up with
tau = 0.8  ## how much the step will be decreased at each iteration

while f.fprime_x(x[0]) != 0:  ##loop to find 0 point for derivative of a function on x

    """checking if new delta is good for x by Armijo–Goldstein condition"""
    step = 0.3
    gradient = np.array(f.fprime_x(x[0]), f.fprime_y(x[1]))
    p = -gradient / ((gradient ** 2).sum() ** 0.5)
    m = gradient.dot(p)
    t = - c * m
    while f.func(*x) - f.func(*(x + step * p)) < step * t:  # good enough step size found
        step *= tau

    """making a step"""

    fx = -f.fprime_x(x)
    x = x + (step * fx)
    print(x[0])

while f.fprime_y(y[0]) != 0:  ##loop to find 0 point for derivative of a function on x

    """checking if new delta is good for y by Armijo–Goldstein condition"""
    step = 0.3
    gradient = np.array(f.fprime_x(x[0]), f.fprime_y(x[1]))
    p = -gradient / ((gradient ** 2).sum() ** 0.5)
    m = gradient.dot(p)
    t = - c * m
    while f.func(*y) - f.func(*(y + step * p)) < step * t:
        step *= tau

    """making a step"""

    fy = -f.fprime_y(y)
    y = y + (step * fy)
    print(x[0], y[0])

print(f"\n\nMIN ({x[0]}, {y[0]})")
f.graph(x[0], y)
