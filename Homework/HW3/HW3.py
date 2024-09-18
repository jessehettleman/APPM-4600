import numpy as np
import math
import matplotlib.pyplot as plt

# Question 1

def question1():
    print("-----Question 1------")

    x = np.linspace(-1,math.pi,100)
    y_1 = np.array([2*X - 1 for X in x])
    y_2 = np.array([math.sin(X) for X in x])

    plt.plot(x,y_1)
    plt.plot(x,y_2)
    plt.savefig("HW3.1.a.png")
    plt.clf()

# use routines  

    # Adjust inputs here here
    f = lambda x: (2*x) - 1 - math.sin(x)
    a = 0
    b = math.pi/2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-8

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('number of iterations = ', count)




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
      
# question1()

# Question 2

def question2():
    print("-----Question 2-----")

# use routines  
    print("Function 1:")
    # Adjust inputs here here
    f = lambda x: (x-5)**9
    a = 4.82
    b = 5.2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-4

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('f(a), f(b) =', f(a), f(b))
    print('number of iterations = ', count)

    print("Function 2 (Expanded Version):")
    # Adjust inputs here here
    f = lambda x: x**9 - 45*(x**8) + 900*(x**7) - 10500*(x**6) + 78750*(x**5) - 393750*(x**4) + 1312500*(x**3) -2812500*(x**2) + 315625*x - 1953125
    a = 4.82
    b = 5.2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-4

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('f(a), f(b) =', f(a), f(b))
    print('number of iterations = ', count)

# question2()

# Question 3

def question3():
    print("-----Question 3-----")

# use routines  
    print("Function 1:")
    # Adjust inputs here here
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-3

    n = math.ceil(math.log2((b-a)/tol) - 1)
    print('expected upper bound on number of iterations: ', n)

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('f(a), f(b) =', f(a), f(b))
    print('number of iterations = ', count)

    print("Function 2 (Expanded Version):")
    # Adjust inputs here here
    f = lambda x: x**9 - 45*(x**8) + 900*(x**7) - 10500*(x**6) + 78750*(x**5) - 393750*(x**4) + 1312500*(x**3) -2812500*(x**2) + 315625*x - 1953125
    a = 4.82
    b = 5.2

# question3()

# Question 5

def question5():
    print("-----Question 5-----")
    # function
    f1 = lambda x: x - (4*math.sin(2*x)) - 3

    x = np.linspace(-2,8,200)
    y = np.array([f1(X) for X in x])

    plt.plot(x,y)
    plt.plot(x,[0 for X in x])
    plt.savefig("HW3.5.a.png")
    plt.clf()

    


    f2 = lambda x: -np.sin(2*x) + ((5*x)/4) - (3/4)
    f2_deriv = lambda x: -2*math.cos(2*x) + (5/4)

    Nmax = 100
    tol = 0.5*(10**(-10))

# test f1 '''
    x0 = 4.6
    [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f2(xstar))
    print('f1_deriv(x0):', f2_deriv(x0))
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
        if (abs(x1-x0) < tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

# question5()