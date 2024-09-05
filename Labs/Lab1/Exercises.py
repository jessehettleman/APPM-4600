import numpy as np
import matplotlib.pyplot as plt

# Pre-Exercises

x = [1,2,3]
x * 3

y = np.array([1,2,3])
y * 3
print('this is 3y', 3*y)

X = np.linspace(0,2*np.pi,100)
Ya = np.sin(X)
Yb = np.cos(X)

# plt.plot(X,Ya)
# plt.plot(X,Yb)
#plt.show()

# plt.plot(X,Ya)
# plt.plot(X,Yb)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# Exercises 3.2

# Problem 1
x = np.linspace(0,10,11)
y = np.arange(0,11,1)

# Problem 2
print(x[0:3])

# Problem 3
print("The first three entries of x are ", x[0], " ", x[1], " ", x[2])

# Problem 4
w = 10**(-np.linspace(1,10,10))
print(w)
# The entries of `w` are fractions exponentially decreasing by a factor of 10 for each integer step.

x = np.arange(1,11,1)
# plt.semilogy(x,w)
# plt.xlabel('x')
# plt.ylabel('w (logarithmic)')
# plt.show()

# Problem 5

s = 3 * w
# print(s)
plt.semilogy(x,w)
plt.semilogy(x,s)
plt.xlabel('x')
plt.ylabel('w, s (logarithmic)')
# plt.show()
plt.savefig("semilogy.png")

# Exercises 4.2

import numpy as np
import numpy.linalg as la
import math

def driver():
    n = 2
    x = np.linspace(0,1,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.

    f = lambda x: x
    g = (1,0)
    y = f(x)
    w = g

    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)

    # print the output
    print('the dot product is : ', dp)
    return

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

driver()


# Problem 1

# I changed the vectors to (0,1) and (1,0)

# Problem 2

def matrixMult(x,y,n):
    # Computes the matrix multiplication product of two SQUARE matrices
    # Initialize the result matrix with zeros
    Mult = [[0 for _ in range(n)] for _ in range(n)]
    
    # Perform matrix multiplication
    for i in range(n):
        for j in range(n):
            for k in range(n):
                Mult[i][j] += x[i][k] * y[k][j]
    
    return Mult

A = [[1,3],[5,6]]
B = [[2,4],[7,2]]

print(matrixMult(A,B,2))
