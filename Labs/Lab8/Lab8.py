import numpy as np
import math
from numpy.linalg import inv 
import matplotlib.pyplot as plt

def line_eval(apts,bpts,xloc):
  x1 = apts[0]
  fx1 = apts[1]

  x2 = bpts[0]
  fx2 = bpts[1]

  m = (fx2 - fx1) / (x2 - x1)
  line = lambda x: fx1 + m*(x - x1)

  return line(xloc)

def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for j in range(Nint):
      '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
      '''let ind denote the indices in the intervals'''
      atmp = xint[j]
      btmp= xint[j+1]
      # find indices of values of xeval in the interval
      ind = np.where((xeval >= atmp) & (xeval <= btmp))
      xloc = xeval[ind]
      n = len(xloc)
      '''temporarily store your info for creating a line in the interval of
      interest'''
      fa = f(atmp)
      fb = f(btmp)
      yloc = np.zeros(len(xloc))

      a = np.array([atmp,fa])
      b = np.array([btmp,fb])


      for kk in range(n):
      #use your line evaluator to evaluate the spline at each location
        yloc[kk] = line_eval(a,b,xloc[kk])

      # Copy yloc into the final vector
      yeval[ind] = yloc

    return yeval

def question1():
    
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 1000
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    plt.plot(xeval,fex,label = 'Actual')
    plt.plot(xeval,yeval,label = 'Line Spline')
    plt.title("Actual vs. Line Spline Interpolation")
    plt.legend()
    plt.show()   
    plt.clf()
     
     
    err = abs(yeval-fex)
    plt.plot(xeval,err)
    plt.title("Plot of Absolute Error")
    plt.show()       

# question1()

# 3.2. The linspline does not perform the strongest interms of absolute error. The maximum error is of the magnitude 10e-1, which occurs near the center of the interval. This is actually the opposite pattern we observed in Lab 7 where we saw "smile" shaped log error curves. This error curve might be particular to this function specifically, however, because the function has a steep spike toward the center of the interval. This means the equispaced linear spline interpolations are fairly erroneous during this steep spike.
    
# def create_natural_spline(yint,xint,N):

# #    create the right  hand side for the linear system
#     b = np.zeros(N+1)
# #  vector values
#     h = np.zeros(N+1)
#     h[0] = xint[1]-xint[0]  
#     for i in range(1,N):
#        h[i] = xint[i+1] - xint[i]
#        b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

#     b[0] = 0
#     b[N] = 0

#     # Create the matrix A so you can solve for the M values
#     A = np.zeros((N+1, N+1))
#     A[0, 0] = 1  # Natural spline boundary condition at x_0
#     A[N, N] = 1  # Natural spline boundary condition at x_N

#     for i in range(1, N):
#         A[i, i-1] = h[i-1] / 6
#         A[i, i] = (h[i-1] + h[i]) / 3
#         A[i, i+1] = h[i] / 6
    
#     # Solve for M using A^-1 * b
#     M = np.zeros(N+1)
#     Ainv = inv(A)
#     M = Ainv @ b
    
# #  Create the linear coefficients
#     C = np.zeros(N)
#     D = np.zeros(N)
#     for j in range(N):
#       C[j] = (yint[j+1] - yint[j]) / h[j] - (h[j] / 6) * (M[j+1] - M[j])
#       D[j] = (M[j+1] - M[j]) / (6 * h[j])
#     return(M,C,D)
       
# def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# # Evaluates the local spline as defined in class
# # xip = x_{i+1}; xi = x_i
# # Mip = M_{i+1}; Mi = M_i

#     hi = xip-xi
    
#     # Compute the cubic spline polynomial
#     term1 = Mi * (xip - xeval) ** 3 / (6 * hi)
#     term2 = Mip * (xeval - xi) ** 3 / (6 * hi)
#     term3 = C * (xeval - xi)
#     term4 = D * (xeval - xi) ** 2
    
#     yeval = term1 + term2 + term3 + term4

#     return yeval 
    
    
# def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
#     yeval = np.zeros(Neval+1)
    
#     for j in range(Nint):
#         '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
#         '''let ind denote the indices in the intervals'''
#         atmp = xint[j]
#         btmp= xint[j+1]
        
# #   find indices of values of xeval in the interval
#         ind= np.where((xeval >= atmp) & (xeval <= btmp))
#         xloc = xeval[ind]

# # evaluate the spline
#         yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
# #   copy into yeval
#         yeval[ind] = yloc

#     return(yeval)

# def question2():
    
#     f = lambda x: 1 / (1 + (10*x)**2)
#     a = -1
#     b = 1
    
#     ''' number of intervals'''
#     Nint = 10
#     xint = np.linspace(a,b,Nint+1)
#     yint = f(xint)

#     ''' create points you want to evaluate at'''
#     Neval = 100
#     xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

# #   Create the coefficients for the natural spline    
#     (M,C,D) = create_natural_spline(yint,xint,Nint)

# #  evaluate the cubic spline     
#     yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
    
#     ''' evaluate f at the evaluation points'''
#     fex = f(xeval)
        
#     nerr = np.linalg.norm(fex-yeval)
#     print('nerr = ', nerr)
    
#     plt.figure()    
#     plt.plot(xeval,fex,'ro-',label='exact function')
#     plt.plot(xeval,yeval,'bs--',label='natural spline') 
#     plt.legend
#     plt.show()
     
#     err = abs(yeval-fex)
#     plt.figure() 
#     plt.semilogy(xeval,err,'ro--',label='absolute error')
#     plt.legend()
#     plt.show()
           
# question2()  

import numpy as np
import matplotlib.pyplot as plt

def create_natural_spline(yint, xint, N):
    # Create the right-hand side for the linear system
    b = np.zeros(N+1)
    h = np.zeros(N)
    
    # Compute vector h and b
    for i in range(N):
        h[i] = xint[i+1] - xint[i]
        if h[i] == 0:
            raise ValueError(f"Zero interval detected between x[{i}] and x[{i+1}]. Ensure xint values are distinct.")
    
    for i in range(1, N):
        b[i] = (yint[i+1] - yint[i]) / h[i] - (yint[i] - yint[i-1]) / h[i-1]
    
    # Ensure boundary conditions for natural spline (b[0] = b[N] = 0)
    b[0] = 0
    b[N] = 0
    
    # Create the matrix A
    A = np.zeros((N+1, N+1))
    A[0, 0] = 1  # Natural spline boundary condition at x_0
    A[N, N] = 1  # Natural spline boundary condition at x_N

    for i in range(1, N):
        A[i, i-1] = h[i-1] / 6
        A[i, i] = (h[i-1] + h[i]) / 3
        A[i, i+1] = h[i] / 6
    
    # Solve for M using np.linalg.solve to avoid inversion issues
    M = np.linalg.solve(A, b)  # Solve the system A * M = b
    
    # Create the linear coefficients C and D
    C = np.zeros(N)
    D = np.zeros(N)
    
    for j in range(N):
        # Correctly calculate the C[j] and D[j] coefficients
        C[j] = (yint[j+1] - yint[j]) / h[j] - h[j] * (2*M[j] + M[j+1]) / 6
        D[j] = (M[j+1] - M[j]) / (6 * h[j])
        
    return M, C, D

def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluates the local spline between xi and xip
    hi = xip - xi  # Interval length
    
    if hi == 0:
        raise ValueError(f"Zero interval detected between xi ({xi}) and xip ({xip}).")
    
    # Compute the cubic spline polynomial
    term1 = Mi * (xip - xeval) ** 3 / (6 * hi)
    term2 = Mip * (xeval - xi) ** 3 / (6 * hi)
    term3 = C * (xeval - xi)
    term4 = D * (xeval - xi) ** 2
    
    yeval = term1 + term2 + term3 + term4
    return yeval

def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        atmp = xint[j]
        btmp = xint[j+1]
        
        # Find indices of xeval in the interval (xint[j], xint[j+1])
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        
        # Evaluate the spline locally
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j+1], C[j], D[j])
        yeval[ind] = yloc
    
    return yeval

def question2():
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    
    # Number of intervals
    Nint = 10
    xint = np.linspace(a, b, Nint+1)
    yint = f(xint)

    # Create points to evaluate the spline
    Neval = 100
    xeval = np.linspace(xint[0], xint[Nint], Neval+1)

    # Create the coefficients for the natural spline    
    M, C, D = create_natural_spline(yint, xint, Nint)

    # Evaluate the cubic spline     
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)
    
    # Evaluate the exact function at the evaluation points
    fex = f(xeval)
        
    nerr = np.linalg.norm(fex - yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval, fex, 'ro-', label='Exact function')
    plt.plot(xeval, yeval, 'bs--', label='Natural spline') 
    plt.legend()
    plt.show()
     
    err = abs(yeval - fex)
    plt.figure() 
    plt.semilogy(xeval, err, 'ro--', label='Absolute error')
    plt.legend()
    plt.show()
           
question2()


