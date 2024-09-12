import numpy as np
import math

def problem1():
    print("Problem 1")

# use routines  

    # Adjust inputs here here
    f = lambda x: (x**2) * (x - 1)
    ab = [[.5,2],[-1,.5],[-1,2]]

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-15

    # loop for all a, b vals
    for pair in ab:
        a = pair[0]
        b = pair[1]
        [astar,ier] = bisection(f,a,b,tol)
        print('the approximate root is',astar)
        print('the error message reads:',ier)
        print('f(astar) =', f(astar))




# define routines
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

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
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
    return [astar, ier]
      
problem1()   

# Question Response:
# For the first interval, the algorithm successfully finds the root with no error. For the second interval, the algorithm encounters an error because f(a)*f(b) is positive, so the algorithm cannot coninute. For the third interval, the algorithm successfully find the root at x = 1, but fails to find the root at x = 0. In fact, it is not possible for the bisection algorithm to find the root at x = 0 because the function does not cross the x-axis at this point, both a and b values will always be negative.

def problem2():
    print("Problem 2")
    # use routines  

    # Adjust inputs here here
    fs = [[lambda x: (x - 1) * (x - 3) * (x - 5), 0, 2.4], [lambda x: (x - 1) * (x - 1) * (x - 3), 0, 2], [lambda x: math.sin(x), 0, .1], [lambda x: math.sin(x), .5, math.pi*(3/4)]]

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-5

    # loop for all a, b vals
    for fun in fs:
        f = fun[0]
        a = fun[1]
        b = fun[2]
        [astar,ier] = bisection(f,a,b,tol)
        print('the approximate root is',astar)
        print('the error message reads:',ier)
        print('f(astar) =', f(astar))

problem2()

# Question Response: 
# This output is what I expected. For the first function, we have three roots and given our a and b values we expect to find a root at x = 1. We find this root with the desired accuracy. For the second function we are again looking for the root at x = 1, but this time it is an even degree so f(a)*f(b) will be positive and we will get an error. This is what the code output indicates occured. For the third function, sin(x), we begin with a value for a that is a root itself, so the algorithm recognizes this and returns that value of a. We do this again with a different interval for sin, and we recieve an error because both f(a) and f(b) are positive again. This means the algorithm is unable to compute a root. 

def problem3():
    print("Problem 3")

# test functions 
    f1 = lambda x: x * (1 + ((7-(x**5))/(x**2)))**3

    f2 = lambda x: x - (((x**5) - 7)/(x**2))

    f3 = lambda x: x - (((x**5) - 7)/(5 * (x**4)))

    f4 = lambda x: x - (((x**5) - 7)/(12))

    Nmax = 100
    tol = 1e-10

# test f1 '''
    # x0 = 1
    # [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f1(xstar):',f1(xstar))
    # print('Error message reads:',ier)
    
#test f2 '''
    # x0 = 1
    # [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f2(xstar):',f2(xstar))
    # print('Error message reads:',ier)

#test f3 '''
    x0 = 1
    [xstar,ier] = fixedpt(f3,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f3(xstar))
    print('Error message reads:',ier)

#test f4 '''
    x0 = 1
    [xstar,ier] = fixedpt(f4,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f4(xstar))
    print('Error message reads:',ier)


# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

problem3()

# Question Response:
# The fixed point iteration converges for both (c) and (d), but it does not converge for (a) and (b). This is due to the fact that the derivatives of f in (a) and (b) are greater than 1 at our initial guess x_0. This causes the algorithm to diverge. Conversely, for both fs in (c) and (d), the derivative of f will be less than 1 at the initial guess, which allows the algorithm to converge. 