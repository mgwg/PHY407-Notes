import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import jv # Bessel functions
from scipy.special import dawsn # Dawson's integral
import time 

colors_light = ["plum", "skyblue", "yellowgreen"]
colors_dark = ["deeppink", "royalblue", "lawngreen"]
colors = ["hotpink", "cornflowerblue"]

# ===================================== a =====================================
def simpson(N, a, b, f, args=[]):
    """
    Uses Simpson's rule to compute the integral of f from a to b, with N points
    Same function as in Q1 

    Args:
        N (int): Even integer
        a (float)
        b (float)
        f (function)
        args (list, optional): list of parameters to be input to f
    
    Returns:
        float
    """
    # step size
    h = (b-a)/N 

    # initialize sum of f
    sum_f = f(a, *args) + f(b, *args)
    # compute sum by interating over points
    for k in range(1, N):
        if k % 2: # k is odd
            sum_f += 4*f(a + k*h, *args)
        else: # k is even
            sum_f += 2*f(a + k*h, *args)

    return 1/3*h*sum_f

def trapezoid(N, a, b, f, args=[]):
    """
    Uses Trapezoidal rule to integral of f from a to b with N points
    Adapted from trapezoid.py from textbook

    Args:
        N (int)
        a (float)
        b (float)
        f (function)
        args (list, optional): list of parameters to be input to f

    Returns:
        float
    """
    # step size
    h = (b-a)/N
    # initialize sum of f
    s = 0.5*f(a, *args) + 0.5*f(b, *args)
    # compute sum by interating over points
    for k in range(1,N):
        s += f(a+k*h, *args)

    return h*s

def dawson_integrand(t):
    """Integrand of Dawson function, excluding constant factors
    """
    return np.exp(t**2)

def dawson_simpson(N, x):
    """Computes Dawson function at x using Simpson's rule with N points

    Returns:
        float
    """
    return np.exp(-x**2)*simpson(N, 0, x, dawson_integrand)

def dawson_trapezoid(N, x):
    """Computes Dawson function at x using trapezoid rule with N points

    Returns:
        float
    """
    return np.exp(-x**2)*trapezoid(N, 0, x, dawson_integrand)

# Define N and x
x = 4
N = 8
# Get D(4) computed by scipy
dawson_scipy = dawsn(x)

# Print values
print("Question 2 a)")
print(f"\tSimpson: {dawson_simpson(N, x)}")
print(f"\tTrapezoidal: {dawson_trapezoid(N, x)}")
print(f"\tScipy: {dawson_scipy}")

# ===================================== b =====================================

error_tol = 5e-9        # error tolerance ~ 1e-9
x = 4                   # x at which Dawson function is computed
num_calls = 50          # number of calls to average over for timing
dawson_scipy = dawsn(x) # D(4) computed by scipy

def compute_error(error_tolerance, reference_value, function, args=[]):
    """Determines the number of points needed to approximate a 
    numerical calculation to less than error_tolerance against a reference. 

    Note that the number of points must be the first argument of the function

    Args:
        error_tol (float): maximum error tolerance
        reference_value (float): reference to compare to
        function (function): function which performs the numerical calculation
        args (array, optional): additional arguments to pass to function

    Returns:
        (float, float): returns the number of points needed and the absolute
        difference between the function output and the reference value
    """
    # Inialize error to an arbitrarily large value
    error = 1000*reference_value 
    # Initialize N such that it will be 2 during the first iteration of the loop
    N = 1

    while error >= error_tol:
        # Increase N by a factor of 2
        N *= 2
        error = np.abs(reference_value - function(N, *args))
    
    return N, error

def time_function(num_calls, function, args):
    """Returns the run time of a function

    Args:
        num_calls (int): number of calls to average run time over
        function (function): function to time
        args (array, optional): additional arguments to pass to function

    """
    time_start = time.time()
    for i in range(num_calls):
        function(*args)
    time_avg = (time.time() - time_start)/num_calls
    
    return time_avg


N_simpson, error_simpson = compute_error(error_tol, dawson_scipy, dawson_simpson, [x])
N_trapezoid, error_trapezoid= compute_error(error_tol, dawson_scipy, 
                                            dawson_trapezoid, [x])

time_simpson = time_function(num_calls, dawson_simpson, [N_simpson, x])
time_trapezoid = time_function(num_calls, dawson_trapezoid, [N_trapezoid, x])

# Print outputs
print("Question 2 b)")

print(f"\tSimpson")
print(f"\t\tvalue:{dawson_simpson(N_simpson, x)}")
print(f"\t\tN = {N_simpson}")
print(f"\t\terror = {error_simpson}")
print(f"\t\ttime = {time_simpson} s")

print(f"\tTrapezoidal")
print(f"\t\t{dawson_trapezoid(N_trapezoid, x)}")
print(f"\t\tN = {N_trapezoid}")
print(f"\t\terror = {error_trapezoid}")
print(f"\t\ttime = {time_trapezoid}")

# ===================================== c =====================================
N1 = 32 
N2 = 64

I1_simpson = dawson_simpson(N1, x)
I2_simpson = dawson_simpson(N2, x)
eps2_simpson = np.abs(1/3*(I2_simpson - I1_simpson))

I1_trapezoid = dawson_trapezoid(N1, x)
I2_trapezoid = dawson_trapezoid(N2, x)
eps2_trapzoid = np.abs(1/15*(I2_trapezoid - I1_trapezoid))

print("Question 2 c)")
print(f"\tSimpson: {eps2_simpson}")
print(f"\tTrapezoidal: {eps2_trapzoid}")