import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm

def monomial(x,y):
    
    # Construct Vandermonde
    n = len(x)
    V = np.zeros((n, n))
    y = np.array(y)

    count = 0
    for xi in x:
        V[count] = np.array([xi**i for i in range(n)])
        count += 1

    poly = np.linalg.solve(V,y)
    return poly

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       

def question1():

    f = lambda x: 1 / (1 + ((10*x)**2))
    xeval = np.linspace(-1,1,1001)
    y = np.array([f(xi) for xi in xeval])

    # Monomial

    mono = monomial(xeval,y)
    err_mono = np.array([abs(y-m) for m in mono])
    
    plt.plot(xeval,mono)
    plt.plot(xeval,y)
    plt.title("Monomial")
    plt.show()
    plt.clf()

    plt.plot(xeval,err_mono)
    plt.title("Monomial Error")
    plt.show()
    plt.clf()

    # Lagrange and Newton DD

    N = 10
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.array([-1 + (j-1)*(2/9) for j in range(11)])
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--') 
    plt.plot(xeval,yeval_dd,'c.--')
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.legend()
    plt.show() 

question1()
