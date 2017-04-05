"""
sample code for finite element method solving
-u''(x) = f(x); u(0) = 0; u(1) = 0, x in [0,1]
This code in not very efficient computationaly.
It serves to illustrate the finite element method
in practice.
"""
from matplotlib import rc
import matplotlib.pylab as plt
from sympy import diff,Symbol,sin, lambdify, pi
import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
from numpy.linalg import inv
from scitools.std import movie
#from scipy import *
import moviepy.editor as mp
N = 3
x = Symbol('x')
def phi(x,k):
    """
    basis functions
    """
    return sin( (k+1)*pi*x )


def f(x):
    """
    right hand side of
    -u''(x) = f(x)
    """
    return x
    
    
def u_e(x):
    """
    Exact solution of 
    -u''(x) = f(x); u(0)=0; u(1)=0
    x in [0,1]
    """
    return (-1./6)*x**3+(1./6)*x


def solution(N):
    A = np.zeros((N,N))
    b = np.zeros(N)
    for i in range(N):
        for j in range(N):
            diff_phi_i = lambdify(x, diff(phi(x,i)))
            diff_phi_j = lambdify(x, diff(phi(x,j)))
            A[i,j]= quad(lambda x: diff_phi_i(x)*diff_phi_j(x), 0,1)[0]
            b[j] = quad(lambda x: phi(x,j)*f(x), 0,1)[0]
        
    #compute C and the numerical solution u(x)
    C = inv(A).dot(b)   
    u = sum([C[i]*phi(x,i) for i in range(N) ])
    u = lambdify(x,u,'numpy')
    return u

#plot solution 
def plot_solution(N):
    z = np.linspace(0,1,N)
    ze = np.linspace(0,1,1000)
    u = solution(N)
    plt.plot(ze,u_e(ze),z,u(z))
    plt.legend(['exact', 'numerical'])
    plt.title('Exact solution vs numerical solution for N={}'.format(N))
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.savefig('figures/{}'.format(N))
    plt.cla()
    plt.clf()
    
    #plt.show()
#for N in range(1,21):
    #plot_solution(N)
    
    
def animate():
    movie('figs/*.png',fps=1,output_file='vid.gif')
    clip = mp.VideoFileClip("vid.gif")
    clip.write_videofile("vid.mp4")

animate()

