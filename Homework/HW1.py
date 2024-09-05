import numpy as np
import matplotlib.pyplot as plt
import math

# -----Problem 1-----

# Coefficient Method
x = np.arange(1.920,2.080,.001)
y_coefficients = x**9 - 18*(x**8) + 144*(x**7) - 672*(x**6) + 2016*(x**5) - 4032*(x**4) +5376*(x**3) - 4608*(x**2) + 2304*x - 512
plt.plot(x,y_coefficients)
plt.savefig("HW1.1.i.png")
plt.clf()

# Binomial Method
y_binom = (x - 2)**9
plt.plot(x,y_binom)
plt.savefig("HW1.1.ii.png")
plt.clf()

# -----Problem 3-----

# Define f and P_2
f = lambda x: (1 + x + x**3)*math.cos(x)
P_2 = lambda x: 1 + x - ((x**2)/2)

f_5 = f(0.5)
P_5 = P_2(0.5)

err_5_bound = 0.5**4
err_5_act = f_5 - P_5

P_int = lambda x,y: x + ((x**2)/2) - ((x**3)/6) - (y + ((y**2)/2) - ((y**3)/6))

# -----Problem 4-----

a = 1
b = -56
c = 1

# Quadratic Formula Method
r_sqrt_round = round(math.sqrt((b**2) - (4*a*c)),3)
r_sqrt_actual = math.sqrt((b**2) - (4*a*c))

r1_quad_round = (-b + r_sqrt_round) / (2*a)
r2_quad_round = (-b - r_sqrt_round) / (2*a)

r1_actual = (-b + r_sqrt_actual) / (2*a)
r2_actual = (-b - r_sqrt_actual) / (2*a)

r1_re = abs((r1_actual - r1_quad_round) / r1_actual) * 100
r2_re = abs((r2_actual - r2_quad_round) / r2_actual) * 100

print("Quadratic Method Error: ", (r1_re,r2_re))

# Root Rearrange Method

# (-r_1 - r_2) = b
r1_rearrange_b = -b - r2_actual
r2_rearrange_b = -b - r1_actual

r1_re_b = abs((r1_actual - r1_rearrange_b) / r1_actual) * 100
r2_re_b = abs((r2_actual - r2_rearrange_b) / r2_actual) * 100

print("Rearrange b Equation Error: ", (r1_re_b,r2_re_b))

# r_1 * r_2 = c
r1_rearrange_c = c / r2_actual
r2_rearrange_c = c / r1_actual

r1_re_c = abs((r1_actual - r1_rearrange_c) / r1_actual) * 100
r2_re_c = abs((r2_actual - r2_rearrange_c) / r2_actual) * 100

print("Rearrange c Equation Error: ", (r1_re_c,r2_re_c))


# -----Problem 5-----

# Part a

# Define the exact values
x1_small = 0.25
x2_small = 0.2499999

x1_large = 10e5 + 1
x2_large = 10e5

# Define the perturbations
delta_x1 = 1e-16
delta_x2 = -1e-16

# Compute the exact and approximate differences
y_small = x1_small - x2_small
y_large = x1_large - x2_large

y_small_approx = (x1_small + delta_x1) - (x2_small + delta_x2)
y_large_approx = (x1_large + delta_x1) - (x2_large + delta_x2)

# Compute the absolute and relative errors
delta_y_small = y_small_approx - y_small
delta_y_large = y_large_approx - y_large

abs_error_small = abs(delta_y_small)
abs_error_large = abs(delta_y_large)

rel_error_small = abs_error_small / abs(y_small)
rel_error_large = abs_error_large / abs(y_large)

# Part b

# Function to compute direct difference and modified expression
def cos_diff_direct(x, delta):
    return np.cos(x + delta) - np.cos(x) # This function takes in an x value and a perturbation delta and returns the cosine difference

def cos_diff_manip(x, delta):
    return -2 * np.sin((2*x + delta) / 2) * np.sin(delta / 2) # This function uses a trigonometric identity to avoid subtraction

# Define x values
x_values = [np.pi, 1e6]

# Define delta values
delta_values = np.logspace(-16, 0, num=16) # This function uses a numpy function to create a array containing a exponential base 10 scale accoring to the exponents listed in the parameters

# Plotting
plt.figure(figsize=(10, 5))

for x in x_values:
    direct_diff = [cos_diff_direct(x, delta) for delta in delta_values] # These arrays calculate the outputs for each delta perturbation value
    manip_diff = [cos_diff_manip(x, delta) for delta in delta_values]

    plt.loglog(delta_values, np.abs(np.array(direct_diff) - np.array(manip_diff)), label=f'x = {x}')
    
plt.xlabel('Delta')
plt.ylabel('Absolute Difference')
plt.title('Difference between Direct Subtraction and Manipulated Expression')
plt.legend()

plt.savefig("HW1.5.b.png")
plt.clf

# Part c

# Function using Taylor expansion
def cos_diff_taylor(x, delta):
    return -delta * np.sin(x) + (delta ** 2 / 2) * (-np.cos(x))

# Compare Taylor expansion with other methods
plt.figure(figsize=(10, 5))

for x in x_values:
    taylor_diff = [cos_diff_taylor(x, delta) for delta in delta_values]
    manip_diff = [cos_diff_manip(x, delta) for delta in delta_values]

    plt.loglog(delta_values, np.abs(np.array(taylor_diff) - np.array(manip_diff)), label=f'x = {x}')

plt.xlabel('Delta')
plt.ylabel('Absolute Difference')
plt.title('Difference between Taylor Expansion and Manipulated Expression')
plt.legend()

plt.savefig("HW1.5.c.png")