import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import matplotlib.pyplot as plt

def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 4*(x[0]**2) + 4*(x[1]**2) - 4
    F[1] = x[0] + x[1] - math.sin(x[0] - x[1])
    return F
    
def evalJ(x): 

    
    J = np.array([[8*x[0], 8*x[1]], 
        [1 - math.cos(x[0] - x[1]), 1 + math.cos(x[0] - x[1])]])
    return J

def SlackerNewton(x0,tol,Nmax):

    ''' Slacker Newton = use the inverse of the Jacobian for initial guess and update conditionally'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    estimates = [x0]

    J = evalJ(x0)
    Jinv = inv(J)

    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        estimates.append(x1)
       
        if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]

        # add slacker condition here
        if its % 3 == 0:
            J = evalJ(x1)
            Jinv = inv(J)
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]  

def prelab():
    h = 0.01*2.**(-np.arange(0, 25))

    f = lambda x: math.cos(x)

    fwd_dif = lambda s,x: (f(s+x) - f(s)) / x
    cnt_dif = lambda s,x: (f(s+x) - f(s-x)) / (2*x)

    fwd = np.array([fwd_dif(math.pi/4,H) for H in h])
    cnt = np.array([cnt_dif(math.pi/4,H) for H in h])

    fwd_err = np.abs(fwd - (-0.70710678118))
    cnt_err = np.abs(cnt - (-0.70710678118))

    print("\nForward Estimation:\n")
    print(fwd)
    print("\nForward Error:\n")
    print(fwd_err)
    print("\nCenter Estimation:\n")
    print(cnt)
    print("\nCenter Error:\n")
    print(cnt_err)

    x_vals = [math.log(H) for H in h]# np.arange(1,11,1)

    # plt.plot(x_vals,[math.log(fwd) for fwd in fwd_err],label='Forward')
    # plt.plot(x_vals,[math.log(cnt) for cnt in cnt_err],label='Center')
    plt.loglog(h,fwd_err,label='Forward')
    plt.loglog(h,cnt_err,label='Center')
    plt.legend()
    plt.title("Log Error of Forward and Center Estimation")
    plt.show()

# prelab()

# These techniques result in the exact same estimations for the slope.
# The forward technique is sublinear, while center is superlinear. This is based on the slopes of the log-log error lines.
# Interestingly, once values of h become very small, approximately  10^-8 for forward method and 10^-5 for center method, the error of each method begins to increase. 

def question1():

    x0 = np.array([1,0])
    
    Nmax = 100
    tol = 1e-10
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  SlackerNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Slacker Newton: the error message reads:',ier)
    print('Slacker Newton: took this many seconds:',elapsed/20)
    print('Slacker Newton: number of iterations is:',its)

question1()

# My implementation of the slacker newton converges to (0.99457618, -0.1040107) within 5 iterations.
# Some of my lab partner's code converged faster. This is because their condition for updating the jacobian was a threshold for the norm of the difference between x1 and x0, which essentially made their code regular Newton.


