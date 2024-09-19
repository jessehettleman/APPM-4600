import numpy as np
import math

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
          return [xstar,ier,approx,count+1]
       x0 = x1
       count = count +1

    xstar = x1
    ier = 1
    return [xstar, ier, approx, count]

# Exercise 2.1

# def order(p,p_hat):
#     order = 0

#     count = 1

#     err_1 = np.zeros((len(p_hat),1))
#     err_2 = np.zeros((len(p_hat),1))

#     while count < len(p_hat):
#         err_ratio_1 = abs(p - p_hat[count]) / abs(p - p_hat[count - 1])
#         err_ratio_2 = abs(p - p_hat[count]) / (abs(p - p_hat[count - 1]))**2
#         err_1[count-1] = err_ratio_1
#         err_2[count-1] = err_ratio_2
#         count += 1

#     return log_err

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)

    print('The order of the equation is:')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))

    return [fit, diff1, diff2]



def exercise2_1():
    f = lambda x: (10/(x+4))**(1/2)


    Nmax = 100
    tol = 1e-10

# test f1 '''
    x0 = 1.5
    p = 1.3652300134140976
    [xstar,ier,approx,count] = fixedpt(f,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f(xstar))
    print('Error message reads:',ier)
    print('Number of iterations:',count)
    print(approx)
    compute_order(approx,p)

# exercise2_1()

# 2.2.a It takes 12 iterations to converge.
# 2.2.b The order of convergence is sublinear because it is associate with an alpha level of approximately 0.88. This means the algorithm is converging slower than linear order.

# 3.2.a p = (p_n+1 ^2 - p_n+2 * p_n) / (2p_n+1 - p_n+2 - P_n)

def atkins(approx,tol,Nmax):
    atkins = np.zeros((len(approx),1))
    count = 0

    while count < Nmax - 2:
        p_n = approx[count]
        p_n1 = approx[count + 1]
        p_n2 = approx[count + 2]

        p = (p_n**2 - p_n2*p_n) / (2*p_n1 - p_n2 - p_n)

        atkins[count] = p

        count += 1

    return atkins

def exercise3_2():
    f = lambda x: (10/(x+4))**(1/2)

    Nmax = 100
    tol = 1e-10

# test f1 '''
    x0 = 1.5
    p = 1.3652300134140976
    [xstar,ier,approx,count] = fixedpt(f,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f(xstar))
    print('Error message reads:',ier)
    print('Number of iterations:',count)
    compute_order(approx,p)

    new = atkins(approx,tol,count)
    print('Atkins Order:')
    compute_order(new,p)

exercise3_2()

# 3.2.c. Now the atkins order of convergence is 0.90, which is faster than before! This means the atkins method improved the order of convergence. 

# 3.4.1 Pseudocode:
# Inputs: p_n, g(x), tol, Nmax
# Outputs: vector of approximations to test convergence
# initialize empty vector
# a = p_n, b = g(a), c = g(b)
# p_n1 = a - (b-a)**2 / (c - 2b + a)
# append p_n1
# return vector

g = lambda x: (10/(x+4))**(1/2)

def steff(p_n, g, tol, Nmax):

    steff = np.zeros((Nmax,1))

    count = 0

    steff[count] = p_n

    count += 1

    while count < Nmax:

        a = p_n
        b = g(a)
        c = g(b)

        p_n1 = a - (((b-a)**2) / (c - 2*b + a))

        steff[count] = p_n1

        p_n = p_n1
        count += 1

    return steff

def exercise3_4():
    g = lambda x: (10/(x+4))**(1/2)
    p = 1.5
    Nmax = 15

    print(steff(p,g,10,Nmax))

exercise3_4()



