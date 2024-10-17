import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
import math

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V   


def eval_lagrange_bary(xeval,xint,yint,N,f):

    # lj = np.ones(N+1)
    
    # for count in range(N+1):
    #    for jj in range(N+1):
    #        if (jj != count):
    #           lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    # yeval = 0
    
    # for jj in range(N+1):
    #    yeval = yeval + yint[jj]*lj[jj]

    phi = 1

    for xi in xint:
        if xi != xeval:
            phi *= (xeval - xi)

    summation = 0

    for j in range(N+1):

        wj_denom = 1

        for i in range(N+1):
            if xint[j] != xint[i]:
                wj_denom *= (xint[j] - xint[i])
            
        wj = 1/wj_denom

        summation += (wj/(xeval-xint[j]))*f(xint[j])
    
    yeval = phi*summation

  
    return(yeval)


def question1(): 

    # Monomial

    f = lambda x: 1 / (1 + ((10*x)**2))
    
    N = 10
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    xint = np.array([-1 + (j-1)*(2/(N-1)) for j in range(1,N+2)])

    '''Create interpolation data'''
    yint = f(xint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)

    # -----1.a-----

    c = Vinv@yint

    # -----1.b-----

    plt.figure(figsize=(8,6))

    Ns = [2,3,19]

    for N in Ns:
        a = -1
        b = 1
        
        ''' Create interpolation nodes'''
        xint = np.array([-1 + (j-1)*(2/(N-1)) for j in range(1,N+2)])
        yint = f(xint)

        ''' Create the Vandermonde matrix'''
        V = Vandermonde(xint,N)

        ''' Invert the Vandermonde matrix'''    
        Vinv = inv(V)

        c = Vinv@yint

        xeval = np.linspace(-1,1,1001)
        # yeval = f(xeval)
        yeval = eval_monomial(xeval,c,N,1000)
        ymax = np.max(yeval)
        
        

        plt.plot(xint,yint,'o')
        plt.plot(xeval,yeval,label=f'N = {N}, max(p(x)={ymax}')

    
    
    plt.legend()
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.savefig("HW7.1.b.png")

    fex = f(xeval)
    plt.figure() 
    err_m = abs(yeval-fex)
    plt.semilogy(xeval,err_m,'ro--',label='monomial error')
    plt.legend()
    plt.savefig("HW7.1.c.png")

    return

question1()
    
def question2():


    # Lagrange Barycentric

    f = lambda x: 1 / (1 + ((10*x)**2))

    plt.figure()   

    Ns = [2,3,19,25,51]

    a = -1
    b = 1

    for N in Ns:
        
        ''' Create interpolation nodes'''
        xint = np.array([-1 + (j-1)*(2/(N-1)) for j in range(1,N+2)])

        '''Create interpolation data'''
        yint = f(xint)
    
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval_l= np.zeros(Neval+1)
    
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange_bary(xeval[kk],xint,yint,N,f)
            

        plt.plot(xeval,yeval_l,label=f'N = {N}') 

    ''' create vector with exact values'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    fex = f(xeval)
    
    plt.plot(xeval,fex,'r-',label='Actual')
    
    plt.legend()
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.savefig("HW7.2.a.png")

    plt.figure() 
    err_l = abs(yeval_l-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange error')
    plt.legend()
    plt.savefig("HW7.2.b.png")
    
    return

question2()

def question3a():

    # Monomial

    f = lambda x: 1 / (1 + ((10*x)**2))
    


    plt.figure(figsize=(8,6))

    Ns = [2,3,19]

    a = -1
    b = 1

    for N in Ns:
        
        ''' Create interpolation nodes'''
        xint = np.array([math.cos(((2*j -1)*math.pi)/(2*N)) for j in range(1,N+2)])
        yint = f(xint)

        ''' Create the Vandermonde matrix'''
        V = Vandermonde(xint,N)

        ''' Invert the Vandermonde matrix'''    
        Vinv = inv(V)

        c = Vinv@yint

        xeval = np.linspace(-1,1,1001)
        yeval = f(xeval)
        yeval = eval_monomial(xeval,c,N,1000)
        fex = f(xeval)
        ymax = np.max(yeval)
        
        

        plt.plot(xint,yint,'o')
        plt.plot(xeval,yeval,label=f'N = {N}, max(p(x)={ymax}')

    
    
    plt.legend()
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.savefig("HW7.3.a.i.png")

    plt.figure() 
    err = abs(yeval-fex)
    plt.semilogy(xeval,err,'ro--',label='monomial error')
    plt.legend()
    plt.savefig("HW7.3.a.ii.png")

# 3a fails due to singular matrix being non-invertivle

def question3b():

    # Lagrange Barycentric

    f = lambda x: 1 / (1 + ((10*x)**2))

    plt.figure()   

    Ns = [2,3,17,18,19]

    a = -1
    b = 1

    for N in Ns:
        
        ''' Create interpolation nodes'''
        xint = np.array([math.cos(((2*j -1)*math.pi)/(2*N)) for j in range(1,N+2)])

        '''Create interpolation data'''
        yint = f(xint)
    
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval_l= np.zeros(Neval+1)
    
        ''' evaluate lagrange poly '''
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange_bary(xeval[kk],xint,yint,N,f)
            

        plt.plot(xeval,yeval_l,label=f'N = {N}') 

    ''' create vector with exact values'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    fex = f(xeval)
    
    plt.plot(xeval,fex,'r-',label='Actual')
    
    plt.legend()
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.savefig("HW7.3.b.i.png")

    plt.figure() 
    err = abs(yeval_l-fex)
    plt.semilogy(xeval,err,'ro--',label='lagrange error')
    plt.legend()
    plt.savefig("HW7.3.b.ii.png")

question3b()




