import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# -----Question 1-----

def evalF1(x): 

    F = np.zeros(2)
    
    F[0] = (x[0]**2) + (x[1]**2) - 4
    F[1] = math.e**x[0] + x[1] - 1

    return F
    
def evalJ1(x): 
    
    J = np.array([[2*x[0], 2*x[1]], 
        [math.e**x[0], 1]])
    return J


def Newton1(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ1(x0)
       Jinv = inv(J)
       F = evalF1(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
def LazyNewton1(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ1(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF1(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
    
def Broyden1(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = evalJ1(x0)

    v = evalF1(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF1(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]

def question1():

    print('Question 1:\n')

    f = lambda x,y: 3*(x**2) - (y**2)
    g = lambda x,y: 3*x*(y**2) - (x**3) - 1

    fs = np.array([f,g])

    x0s = np.array([[1, 1],[1, -1],[0, 0]])
    
    Nmax = 100
    tol = 1e-10

    for x0 in x0s:

        print('\nMethods for initial guess:',x0,'\n')

        if x0[0] == 0 and x0[1] == 0:
            print("ERROR: Initial Guess Causes Non-Invertible Jacobian\n")
            break
    
        t = time.time()
        for j in range(50):
            [xstar,ier,its] =  Newton1(x0,tol,Nmax)
        elapsed = time.time()-t
        print(xstar)
        print('Newton: the error message reads:',ier) 
        print('Newton: took this many seconds:',elapsed/50)
        print('Netwon: number of iterations is:',its)
        
        t = time.time()
        for j in range(20):
            [xstar,ier,its] =  LazyNewton1(x0,tol,Nmax)
        elapsed = time.time()-t
        print(xstar)
        print('Lazy Newton: the error message reads:',ier)
        print('Lazy Newton: took this many seconds:',elapsed/20)
        print('Lazy Newton: number of iterations is:',its)
        
        t = time.time()
        for j in range(20):
            [xstar,ier,its] = Broyden1(x0, tol,Nmax)     
        elapsed = time.time()-t
        print(xstar)
        print('Broyden: the error message reads:',ier)
        print('Broyden: took this many seconds:',elapsed/20)
        print('Broyden: number of iterations is:',its)

question1()

# -----Question 2-----

def evalF2(x): 

    F = np.zeros(3)
    
    F[0] = x[0] + math.cos(x[0]*x[1]*x[2]) - 1
    F[1] = (1-x[0])**(1/4) + x[1] + 0.05*(x[2]**2) - 0.15*x[2] - 1
    F[2] = -(x[0]**2) - 0.1*(x[1]**2) + 0.01*x[1] + x[2] - 1
    return F
    
def evalJ2(x): 

    
    J = np.array([[1 - x[1]*x[2]*math.sin(x[0]*x[1]*x[2]), (-1)*x[0]*x[2]*math.sin(x[0]*x[1]*x[2]), (-1)*x[0]*x[1]*math.sin(x[0]*x[1]*x[2])], 
        [(-1/4)*(1-x[0])**(-3/4), 1, 0.1*x[2] - 0.15], 
        [-2*x[0], -0.2*x[1] + 0.01, 1]])
    return J


def Newton2(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ2(x0)
       Jinv = inv(J)
       F = evalF2(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def evalg(x):

    F = evalF2(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF2(x)
    J = evalJ2(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg

def SteepestDescent(x,tol,Nmax):

    count = 0
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            count += 1
            return [x,g1,ier,count]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            count += 1
            return [x,gval,ier,count]

        count += 1

    print('max iterations exceeded')    
    ier = 1       
    return [x,g1,ier,count]

def Hybrid(x0,tol,Nmax):

    steepest_return = SteepestDescent(x0,5e-2,Nmax)
    steepest_guess = steepest_return[0]

    newton_return = Newton2(steepest_guess,tol,Nmax)
    its = steepest_return[3] + newton_return[2]

    return [newton_return[0],newton_return[1],its]

def question2():

    print('Question 2:\n')

    x0 = np.array([-.5, 1.5, .5])
    
    Nmax = 100
    tol = 1e-6

    print("\nNewton Method:\n")
    
    t = time.time()
    for j in range(50):
        [xstar,ier,its] =  Newton2(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)

    print("\nSteepest Descent Method:\n")

    t = time.time()
    for j in range(50):
        [xstar,gval,ier,its] = SteepestDescent(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print("Steepest: g evaluated at this point is:", gval)
    print("Steepest: the error message reads:", ier	)
    print('Steepest: took this many seconds:',elapsed/50)
    print('Steepest: number of iterations is:',its)

    print("\nHybrid Method:\n")

    t = time.time()
    for j in range(50):
        [xstar,ier,its] =  Hybrid(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Hybrid: the error message reads:',ier) 
    print('Hybrid: took this many seconds:',elapsed/50)
    print('Hybrid: number of iterations is:',its)

    print('\n')


    
question2()



