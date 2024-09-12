import numpy as np
import matplotlib.pyplot as plt
import math
import random

# test
# -----Problem 2-----

# Initialize Matrices

A = np.array([[(1/2), (1/2)], [(1/2)*(1+(10**(-10))), (1/2)*(1-(10**(-10)))]])
b = np.array([1,1])
x = np.array([1,1])

cond_A = np.linalg.cond(A) # 1 / (1-(10**(-10)))

A_inv = np.array([[(1-(10**10)), (10**10)], [(1+(10**10)), (-(10**10))]])

# Changed b_delta values to answer part c
b_delta = np.array([5*(10**(-5)), 5*(10**(-5))])

x_tilde = [(A_inv[0][0]*(b_delta[0] + b[0])) + (A_inv[0][1]*(b_delta[1] + b[1])), (A_inv[1][0]*(b_delta[0] + b[0])) + (A_inv[1][1]*(b_delta[1] + b[1]))]

x_relerr = np.linalg.norm(np.matmul(A_inv,b_delta))/np.linalg.norm(x) # [abs((x[0] - x_tilde[0])/x[0]), abs((x[1] - x_tilde[1])/x[1])]

# print("Condition number of A: ", cond_A)
# print("Relative error of x_tilde after perturbation b_delta: ", x_relerr)


# -----Problem 3-----

def alg_1(x):
    y = math.e**(x)
    return y - 1

def cond(x):
    return x * ((math.e ** x)/(alg_1(x)))

xvals = np.arange(-0.05,0.05,0.001)
yvals = np.array([alg_1(x) for x in xvals])

plt.plot(xvals,yvals)
plt.savefig("HW2.3.b.png")
plt.clf()

part_c = (9.999999995000000 * (10**(-10)))

print("Algorith output for value in part c: ", alg_1(part_c))
print("Condition number for value in part c: ", cond(part_c))

def taylor(x,n):

    # Initialize variable for approximation
    approx = 0

    # Adds terms to taylor polynomical. First term in e^x is 1, which is cancelled by the subtraction of 1
    for i in range(1,n+1):
        approx = approx + (x**i)/(math.factorial(i))

    return approx

def relerr(x,n):
    return abs(np.expm1(x) - taylor(x,n))/abs(np.expm1(x))

for n in range(1,10):
    print("Relative error for value in part c given ", n, " terms in series: ", relerr(part_c,n))


# -----Problem 4a-----

t = np.linspace(0, math.pi, 31)
y = np.array([math.cos(T) for T in t])

# Calculate sum and print:
count = 0
S = 0
while count < len(t):
    S += t[count]*y[count]
    count += 1

print("The sum is: ", S)


# -----Problem 4b-----

R = 1.2
delr = 0.1
f = 15
p = 0

theta = np.linspace(0, 2*math.pi, 100)

x = np.array([R * (1 + delr*math.sin(f*th + p)) * math.cos(th) for th in theta])
y = np.array([R * (1 + delr*math.sin(f*th + p)) * math.sin(th) for th in theta])

plt.plot(x,y)
plt.savefig("HW2.4.b.i.png")
plt.clf()

for i in range(1,11):

    R = i
    delr = 0.05
    f = 2 + i
    p = random.uniform(0,2)

    x = np.array([R * (1 + delr*math.sin(f*th + p)) * math.cos(th) for th in theta])
    y = np.array([R * (1 + delr*math.sin(f*th + p)) * math.sin(th) for th in theta])

    plt.plot(x,y)

plt.savefig("HW2.4.b.ii.png")
plt.clf()


