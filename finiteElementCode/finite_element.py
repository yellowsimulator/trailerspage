"""
sample code for finite element
"""
from sympy import *
import numpy as np
def Lagrange_polynomial(x, i, points):
    """
    Return the Lagrange polynomial no. i.
    points are the interpolation points, and x can be a number or
    a sympy.Symbol object (for symbolic representation of the
    polynomial). When x is a sympy.Symbol object, it is
    normally desirable (for nice output of polynomial expressions)
    to let points consist of integers or rational numbers in sympy.
    """
    p = 1
    for k, point in enumerate(points) :
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p


x = Symbol('x')
dx = 0.1
N = 1./dx
xr = np.linspace(0,1,N)
points=list(xr)
ids=[points.index(j) for j in points]
i = ids[0]

phi = Lagrange_polynomial(x, i, points)


A[i,j] = integrate(diff( phi  ))
