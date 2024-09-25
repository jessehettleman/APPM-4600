import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd

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

def secant(x0,x1,f,Nmax,tol):

  approx = np.zeros((Nmax,1))

  count = 0
  x2 = x1

  if abs(f(x1) - f(x0)) == 0:
    ier = 1
    xstar = x1
    approx[count] = x1
    return [xstar, ier,approx,count]

  for i in range(1,Nmax+1):
    x2 = x1 - (f(x1) * ((x1 - x0)/(f(x1) - f(x0))))
    approx[count] = x2
    count += 1
    
    if abs(x2 - x1) < tol:
      xstar = x2
      ier = 0
      return [xstar, ier,approx,count]

    x0 = x1
    x1 = x2

    if(abs(f(x1) - f(x0))) == 0:
      ier = 1
      xstar = x2
      return [xstar, ier,approx,count]

  xstar = x2
  ier = 1
  return [xstar, ier,approx,count]

# Question 1

def question1():
    print("-----Question 1------")

    f = lambda x: scipy.special.erf(x/1.69161697792) - (3/7)

    x = np.linspace(0,2,100)
    y = np.array([f(X) for X in x])

    plt.plot(x,y)
    plt.plot(x, [0 for X in x])
    plt.savefig("HW4.1.a.png")
    plt.clf()

    # Bisection
    print("Bisection Method:")

    a = 0
    b = 2

    tol = 1e-13

    [astar,ier,count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('number of iterations = ', count)

    print("Newton Method:")

    f_deriv = lambda x: (2/(1.69161697792*math.pi))*math.e**((-x/1.69161697792)**2)

    g = lambda x: x - (f(x)/f_deriv(x))
    Nmax = 100
    tol = 1e-6

    x0 = 0.01
    [xstar,ier,approx,count] = fixedpt(g,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('g(xstar):',g(xstar))
    print('Error message reads:',ier)
    print(approx)

# question1()

# Question 4

def question4():

  print("Newton Method:")

  f = lambda x: (math.e**(3*x)) - (27*(x**6)) + (27*(x**4)*(math.e**x)) - (9*(x**2)*(math.e**(2*x)))
  f_deriv = lambda x: (3*math.e**(3*x)) - (162*(x**5)) + (108*(x**3)*(math.e**x)) + (27*(x**4)*(math.e**x)) - (18*(x)*(math.e**(2*x))) - (18*(x**2)*(math.e**(2*x)))

  g = lambda x: x - (f(x)/f_deriv(x))
  Nmax = 100
  tol = 1e-10

  x0 = 4
  [xstar,ier,approx,count] = fixedpt(g,x0,tol,Nmax)
  print('the approximate fixed point is:',xstar)
  print('g(xstar):',g(xstar))
  print('Error message reads:',ier)
  print('number of iterations = ', count)

  print("2c Fix -- Multiply By m Method:")

  m = 3
  g2 = lambda x: x - m*(f(x)/f_deriv(x))
  Nmax = 100
  tol = 1e-10

  x0 = 4
  [xstar,ier,approx,count] = fixedpt(g2,x0,tol,Nmax)
  print('the approximate fixed point is:',xstar)
  print('g(xstar):',g2(xstar))
  print('Error message reads:',ier)
  print('number of iterations = ', count)

  print("g = f/f' Fix Method:")

  f_divf = lambda x: f(x)/f_deriv(x)
  f_divf_deriv = lambda x: ((4374*(x**10)) + (729*(math.e**x)*(x**10)) - (2916*(math.e**x)*(x**9)) - (4374*(math.e**x)*(x**8)) - (972*(math.e**(2*x))*(x**8)) + (3888*(math.e**(2*x))*(x**7)) + (972*(math.e**(2*x))*(x**6)) + (486*(math.e**(3*x))*(x**6)) - (1944*(math.e**(3*x))*(x**5)) - (108*(math.e**(4*x))*(x**4)) + (324*(math.e**(3*x))*(x**4)) + (432*(math.e**(4*x))*(x**3)) + (9*(math.e**(5*x))*(x**2)) - (162**(math.e**(4*x))*(x**2)) - (36*(math.e**(5*x))) + (18*(math.e**(5*x)))) / ((f_deriv(x))**2)

  g3 = lambda x: x - (f_divf(x)/f_divf_deriv(x))
  Nmax = 100
  tol = 1e-10

  x0 = 4
  # [xstar,ier,approx,count] = fixedpt(g3,x0,tol,Nmax)
  # print('the approximate fixed point is:',xstar)
  # print('g(xstar):',g3(xstar))
  # print('Error message reads:',ier)
  # print('number of iterations = ', count)



# question4()

def question5():

  f = lambda x: (x**6) - x - 1
  f_deriv = lambda x: 6*(x**5) - 1

  print("Newton Method:")

  g = lambda x: x - (f(x)/f_deriv(x))

  Nmax = 100
  tol = 1e-10

  x0 = 2
  [xstar,ier,approx,count] = fixedpt(g,x0,tol,Nmax)
  print('the approximate fixed point is:',xstar)
  print('g(xstar):',g(xstar))
  print('Error message reads:',ier)
  print('number of iterations = ', count)
  # print(approx)

  newton_error = [abs(xstar - approx[i]) for i in range(0,count+1)]
  iterations = list(range(1,count))

  Newton = pd.DataFrame(list(zip(iterations, approx, newton_error)))
  Newton.columns = ['Iteration','Estimate','Error']
  print("\nNewton Method Error Table:\n")
  print(Newton)
  print("\n")


  print("Secant Method:")

  Nmax = 100
  tol = 1e-10

  x0 = 2
  x1 = 1
  [xstar,ier,approx,count] = secant(x0,x1,f,Nmax,tol)
  print('the approximate fixed point is:',xstar)
  print('f(xstar):',f(xstar))
  print('Error message reads:',ier)
  print('number of iterations = ', count)
  # print(approx)

  secant_error = [abs(xstar - approx[i]) for i in range(0,count+1)]
  iterations = list(range(1,count))

  Secant = pd.DataFrame(list(zip(iterations, approx, newton_error)))
  Secant.columns = ['Iteration','Estimate','Error']
  print("\nSecant Method Error Table:\n")
  print(Secant)
  print("\n")

  plt.clf()
  plt.figure(figsize=(10, 6))
  plt.loglog(newton_error[:-1], newton_error[1:], 'bo-', label="Newton Method")
  plt.loglog(secant_error[:-1], secant_error[1:], 'ro-', label="Secant Method")
  plt.xlabel(r'$|x_k - \alpha|$', fontsize=14)
  plt.ylabel(r'$|x_{k+1} - \alpha|$', fontsize=14)
  plt.legend()
  plt.grid(True, which="both", ls="--")
  plt.savefig("HW4.5.b.png")


question5()


