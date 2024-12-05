import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
  
     x = scila.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
# Modify the code to include LU-based solving alongside the built-in solver, and timing logic for comparison.


def modified_driver():

     matrix_sizes = [100, 500, 1000, 2000, 4000, 5000]
     num_rhs = [1, 5, 10, 50, 100]  # Different numbers of right-hand sides

     results = {"size": [], "rhs_count": [], "lu_decomp_time": [], "lu_solve_time": [], "builtin_time": []}
     
     for N in matrix_sizes:
     # Generate random square matrix and single right-hand side
          A = np.random.rand(N, N)
          for k in num_rhs:
               B = np.random.rand(N, k)  # Multiple RHS
     
               start = time.time()
               x_builtin = scila.solve(A, B)
               builtin_time = time.time() - start
  
               start = time.time()
               lu, piv = scila.lu_factor(A)  
               lu_decomp_time = time.time() - start
  
               start = time.time()
               x_lu = scila.lu_solve((lu, piv), B)
               lu_solve_time = time.time() - start

               results["size"].append(N)
               results["rhs_count"].append(k)
               results["builtin_time"].append(builtin_time)
               results["lu_decomp_time"].append(lu_decomp_time)
               results["lu_solve_time"].append(lu_solve_time)


     sizes = np.unique(results["size"])
     for k in num_rhs:

        indices = [i for i in range(len(results["rhs_count"])) if results["rhs_count"][i] == k]
        sizes_k = [results["size"][i] for i in indices]
        builtin_times_k = [results["builtin_time"][i] for i in indices]
        total_lu_times_k = [results["lu_decomp_time"][i] + results["lu_solve_time"][i] for i in indices]

        plt.figure()
        plt.plot(sizes_k, builtin_times_k, label="Built-in Solver", marker="o")
        plt.plot(sizes_k, total_lu_times_k, label="LU Solver (Total)", marker="s")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Time (seconds)")
        plt.title(f"Performance Comparison (RHS = {k})")
        plt.legend()
        plt.grid(True)
        plt.show()
               
     print(results)

modified_driver()
