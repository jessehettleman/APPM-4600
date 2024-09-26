import numpy as np
import math

def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 
    count = 0

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]
    
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    approx = np.zeros((Nmax,1))

    count = 0
    while (count <Nmax):
       x1 = f(x0)
       approx[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          count += 1
          return [xstar,ier,approx,count]
       x0 = x1
       count = count +1

    xstar = x1
    ier = 1
    return [xstar, ier,approx,count]

def bisection_new(f,f_deriv,f_deriv_2,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 
    count = 0

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]
    
    d = 0.5*(a+b)
    count += 1
    while ((f(d)*f_deriv_2(d))/((f_deriv(d))**2) < 1): # testing if within basin of convergence
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
        b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

def fixedpt_new(g,f,f_deriv,f_deriv_2,a,b,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    x0,ier,count = bisection_new(f,f_deriv,f_deriv_2,a,b,tol)

    approx = np.zeros((Nmax,1))

    while (count <Nmax):
       x1 = g(x0)
       approx[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          count += 1
          return [xstar,ier,approx,count]
       x0 = x1
       count = count +1

    xstar = x1
    ier = 1
    return [xstar, ier,approx,count]

def driver():
    f = lambda x: math.e**((x**2) + (7*x) - 30) - 1
    f_deriv = lambda x: ((2*x) + 7)*math.e**((x**2) + (7*x) - 30)
    f_deriv_2 = lambda x: 2*math.e**((x**2) + (7*x) - 10) + (((2*x) + 7)**2)*math.e**((x**2) + (7*x) - 30)

    print("\nBisection Method:\n")

    a = 2
    b = 4.5

    tol = 1e-10

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('number of iterations = ', count)

    print("\nNewton Method:\n")

    g = lambda x: x - (f(x)/f_deriv(x))
    Nmax = 100
    tol = 1e-10

    x0 = 4.5
    [xstar,ier,approx,count] = fixedpt(g,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('g(xstar):',g(xstar))
    print('Error message reads:',ier)
    print('number of iterations = ', count)

    print("\nModified Method:\n")

    [xstar,ier,approx,count] = fixedpt_new(g,f,f_deriv,f_deriv_2,a,b,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('g(xstar):',g(xstar))
    print('Error message reads:',ier)
    print('number of iterations = ', count)

driver()


# 1. g(x) = x - f(x)/f'(x), so we want to find when g'(x) < 1. g'(x) = (f(x)*f''(x))/(f'(x))**2 < 1. This will function as our criteria for the while loop within the bisection method, and will return an initial guess that is guaranteed to be within the basin of convergence.
# 2. Yes, the input of the bisection method needed to change because now we need f'(x) and f''(x) in addition to all of the original inputs. Tolerance becomes optional here because we will no longer iterate our while loop upon that information.
# 5. The advantages are that the Newton method is guaranteed to run within its basin of convergence. One of the disadvantages is that we use time to ensure the guess is in the basin of convergence when it already might be. In this case, we would actually have a slightly less efficient method. However, this method is overall more robust because it will eliminate some of the issues Newton experiences with functions that take a long time to reach the basin of convergence.
# 6. The bisection method converged in 34 iterations, the Newton method converged in 27 iterations, and the Modified method took only 10 iterations! Iteration wise, the most efficient algorithm is the modified method. However, this method does involve computing the first and second derivatives beforehand, which could be expensive. 