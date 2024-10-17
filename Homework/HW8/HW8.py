import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque

# Paste All Interpolation Method Functions Here

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def question1():

    # Setup
    f = lambda x: 1 / (1 + x**2)
    a = -5
    b = 5
    Ns = [5,10,15,20]
    errs = deque()

    # -----1.a-----
    # Lagrange Interpolation

    plt.figure()

    for N in Ns:
        
        ''' Create interpolation nodes'''
        xint = np.linspace(a,b,N+1)

        '''Create interpolation data'''
        yint = f(xint)
    
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval_l= np.zeros(Neval+1)
        fex = f(xeval)

    
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            

        plt.plot(xeval,yeval_l,label=f'N = {N}') 

        errs.append(abs(yeval_l-fex))

    err_l = np.array(errs)

    ''' create vector with exact values'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    fex = f(xeval)
    
    plt.plot(xeval,fex,'r-',label='Actual')
    
    plt.legend()
    plt.savefig("HW8.1.a.i.png")

    plt.figure() 
    count = 5
    for err in err_l:
        #err_l = abs(yeval_l-fex)
        plt.semilogy(xeval,err,label=f'N = {count} error')
        count += 5
    plt.legend()
    plt.savefig("HW8.1.a.ii.png")


    #-----1.b-----
    # Hermite Interpolation


    #-----1.c-----
    # Natural Cubic Spline


    #-----1.d-----
    # Clamped Cubic Spline

question1()

def question2():
     # Setup
    f = lambda x: 1 / (1 + x**2)
    a = -1
    b = 1
    Ns = [5,10,15,20]
    errs = deque()

    # -----2.a-----
    # Lagrange Interpolation

    plt.figure()

    for N in Ns:
        
        ''' Create Chebyshev interpolation nodes'''
        xint = np.array([math.cos(((2*j -1)*math.pi)/(2*N)) for j in range(1,N+2)])

        '''Create interpolation data'''
        yint = f(xint)
    
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval_l= np.zeros(Neval+1)
        fex = f(xeval)

    
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            

        plt.plot(xeval,yeval_l,label=f'N = {N}') 

        errs.append(abs(yeval_l-fex))

    err_l = np.array(errs)

    ''' create vector with exact values'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    fex = f(xeval)

    plt.plot(xeval,fex,'r-',label='Actual')
    
    plt.legend()
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.savefig("HW8.2.a.i.png")

    plt.figure() 
    count = 5
    for err in err_l:
        #err_l = abs(yeval_l-fex)
        plt.semilogy(xeval,err,label=f'N = {count} error')
        count += 5
    plt.legend()
    plt.savefig("HW8.2.a.ii.png")


    #-----2.b-----
    # Hermite Interpolation


    #-----2.c-----
    # Natural Cubic Spline


    #-----2.d-----
    # Clamped Cubic Spline

question2()
