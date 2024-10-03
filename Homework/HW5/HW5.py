import numpy as np
import math
import matplotlib.pyplot as plt

def newton_2(h,J,x0,tol,Nmax):
    """
    Newton iteration.

    Inputs:
    f,J - function array and derivative
    n - matrix dimensions
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)
        
    """
    #   x = np.zeros(Nmax+1)
    #   x[0] = x0

    f = h[0]
    g = h[1]

    f_x = J[0][0]
    f_y = J[0][1]
    g_x = J[1][0]
    g_y = J[1][1]

    for it in range(Nmax):
        x,y = x0
        h_x0 = np.array([f(x,y),g(x,y)])
        J_x0 = np.array([[f_x(x,y), f_y(x,y)],[g_x(x,y),g_y(x,y)]])

        pn = np.linalg.solve(J_x0,-h_x0)
        x1 = x0 + pn
        
    #   x[it+1] = x1
        if (np.linalg.norm(x1-x0) < tol):
            xstar = x1
            info = 0
            return [xstar,info,it]
        x0 = x1
    xstar = x1
    info = 1
    return [xstar,info,it]

def question1():

    f = lambda x,y: 3*(x**2) - (y**2)
    g = lambda x,y: 3*x*(y**2) - (x**3) - 1

    fs = np.array([f,g])

    x_y_0 = np.array([1,1])
    J_inv = np.array([[(1/6),(1/18)],[0,(1/6)]])

    x_y_n = x_y_0

    for n in range(0,11):
        for i in range(0,n):
            x,y = x_y_n
            f_g_n = np.array([f(x,y),g(x,y)])

            x_y_n = x_y_n - np.matmul(J_inv,f_g_n)

        print(n,"iterations:", x_y_n)

    J_0 = np.array([[6,-2],[0,6]])
    J_0_inv = np.linalg.inv(J_0)
    print("J(1,1)^-1:",J_0_inv)

    print("\nNewton Method:\n")

    f_x = lambda x,y: 6*x
    f_y = lambda x,y: -2*y
    g_x = lambda x,y: 3*(y**2) - 3*(x**2)
    g_y = lambda x,y: 6*x*y
    J = np.array([[f_x, f_y],[g_x,g_y]])

    Nmax = 100
    tol = 1.e-14

    (xstar,info,it) = newton_2(fs,J,x_y_0,tol,Nmax)
    print('the approximate root is', xstar)
    print('the error message reads:', info)
    print('Number of iterations:', it)

# question1()

def question3(): 

    f = lambda x,y,z: x**2 + 4*y**2 + 4*z**2 - 16

    f_x = lambda x,y,z: 2*x
    f_y = lambda x,y,z: 8*y
    f_z = lambda x,y,z: 8*x

    def newton_step(x, y, z):
        fx = f_x(x, y, z)
        fy = f_y(x, y, z)
        fz = f_z(x, y, z)
        f_val = f(x, y, z)

        d = f_val / (fx**2 + fy**2 + fz**2)

        x_n = x - d * fx
        y_n = y - d * fy
        z_n = z - d * fz

        return np.array([x_n, y_n, z_n])
    
    # Initial Guess
    v = np.array([1, 1, 1])
    true = np.array([1.09214606,1.35232445,1.36858424])
    err = np.linalg.norm(v - true)

    n = 0
    errors = np.zeros((11,1))
    errors[n] = err

    print("\nIterations and Errors:\n")
    print(v,err)

    for i in range(10): 
        n += 1
        v = newton_step(v[0], v[1], v[2])
        err = np.linalg.norm(v - true)
        errors[n] = err
        print(v,err)
    print("\n")

    x_vals = np.array([i for i in range(0,11)])

    plt.plot(x_vals,[math.log(err) for err in errors])
    plt.xlabel("Iterations")
    plt.ylabel("Log Error")
    plt.title("Quadratic Convergence of Iteration Scheme")
    plt.savefig("HW5.3.b.png")


question3()


