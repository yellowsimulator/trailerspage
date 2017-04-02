"""
sample code for finite element method solving
-u''(x) = f(x); u(0) = 0; u(1) = 0, x in [0,1]
"""
import matplotlib.pylab as plt
from sympy import diff,Symbol,sin, lambdify, pi
import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
from numpy.linalg import inv

N = 5
x = Symbol('x')
def phi(x,k):
    """
    basis functions
    """
    return sin( (k+1)*pi*x )


def diff_phi(phi,x,k):
    """
    derivative of basis function
    """
    return diff(phi(x,k),x)


def f(x):
    """
    right hand side of
    -u''(x) = f(x)
    """
    return x
    
    
def u_e(x):
    """
    Exct solution of 
    -u''(x) = f(x); u(0)=0; u(1)=0
    x in [0,1]
    """
    return (-1./6)*x**3+(1./6)*x

#construct matrix A and vector b
A = np.zeros((N,N))
b = np.zeros(N)
for i in range(N):
    for j in range(N):
        diff_phi_i = lambdify(x,diff_phi(phi,x,i))
        diff_phi_j = lambdify(x,diff_phi(phi,x,j))
        A[i,j]= quad(lambda x: diff_phi_i(x)*diff_phi_j(x), 0,1)[0]
        b[j] = quad(lambda x: phi(x,j)*f(x), 0,1)[0]
        
#compute C and the numerical solution u(x)
m = range(N)
C = inv(A).dot(b)   
u = sum([C[i]*phi(x,k) for i, k in zip(m,m) ])
u = lambdify(x,u,'numpy')

#plot solution 
z = np.linspace(0,1,N)
ze = np.linspace(0,1,1000)

plt.plot(ze,u_e(ze),z,u(z))
plt.legend(['exact', 'numerical'])
plt.title('Exact solution vs numerical solution for N={}'.format(N))
plt.xlabel('x')
plt.ylabel('u(x)')
plt.show()






