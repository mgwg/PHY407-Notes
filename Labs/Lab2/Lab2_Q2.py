import numpy as np
import matplotlib.pyplot as plt
from scipy.special import dawsn
import matplotlib
# increase defualt font size
matplotlib.rcParams.update({'font.size': 15})

# Define functions that will be used throught the question 
def simpson(N, a, b, f, args=[]):
    """
    From lab 1: ses Simpson's rule to compute the integral of f from a to b, with N points
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
    From lab 1: uses Trapezoidal rule to integral of f from a to b with N points
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

def gaussxw(N):
    """From week 3 practical. Returns array of points x and weights w for gaussian quadrature
    integration, for integration range from -1 to 1

    Args:
        N (int): number of integration slices

    Returns:
        (array, array): arrays of points and weights
    """
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    """From week 3 practical. Returns array of points x and weights w for gaussian quadrature
    integration, for integration range from a to b

    Args:
        N (int): number of integration slices
        a (float): start of integration interval
        b (float): end of integration interval
    """
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gaussian_quad(N, a, b, f, args=[]):
    """returns integral of f computed using gaussian quadrature, using N points from a to b

    Args:
        N (int): number of integration slices
        a (float): start of integration interval
        b (float): end of integration interval
        f (func): integrand
        args (list, optional): additional argumetns to pass to f

    Returns:
        float
    """
    pos, w = gaussxwab(N, a, b)
    return np.sum(w*f(pos, *args))

def dawson(N, x, method):
    """Returns dawson's function at x, numerically integrated using a specified method. 
    Method may be:
        "trapz" for trapezoidal rule
        "simp" for simpson's rule
        "gauss" for gaussian quadrature


    Args:
        N (int): number of integration slices
        x (float): value to evaluate dawson function at
        method (str): numerical integration method

    Returns:
        float
    """
    integration_function = {"trapz":trapezoid, "simp": simpson, "gauss": gaussian_quad}[method]
    return np.exp(-x**2)*integration_function(N, 0, x, lambda t:np.exp(t**2))

x = 4 # x to evaluate dawson function at
target_nvals = 100 # desired num points on graph

# Get logarithmically spaced array of Ns between 8 and 2048
# Additionally, require all Ns to be ints, and remove duplicate N values after rounding 
N = np.unique(np.logspace(3, 11, target_nvals, base=2).astype(int))
# get actual number of points after rounding
nvals = len(N)

# Select even Ns for Simpson's rule
N_simp = N[N%2 == 0]
nvals_simp = len(N_simp)

# initialize arrays for integrals
trapz = np.zeros(nvals)
simp = np.zeros(nvals_simp)
gauss = np.zeros(nvals)
# also use a separate array to store values at 2N to calculate errors with
gauss2N = np.zeros(nvals)

# calcualte values
for i in range(nvals):
    trapz[i] = dawson( N[i], x, "trapz" )
    gauss[i] = dawson( N[i], x, "gauss" )
    gauss2N[i] = dawson( 2*N[i], x, "gauss" )

for i in range(nvals_simp):
    simp[i] = dawson( N_simp[i], x, "simp" )

# get error on gaussian quadrature
gauss_err = np.abs(gauss2N - gauss)
# and relative error compared to scipy's value
gauss_err_rel = np.abs((gauss - dawsn(4))/dawsn(4))

# make plots for each question
# ================================= a ==============================================

fig, ax = plt.subplots()
ax.plot(N, trapz, color="hotpink", label="trapezoidal rule")
ax.plot(N_simp, simp, color="cornflowerblue", label="simpson's rule")
ax.plot(N, gauss, color="yellowgreen", label="gaussian quadrature")

ax.set_xlabel("N")
ax.set_ylabel(f"D(4)")
ax.legend()
ax.set_xscale("log")
plt.savefig("../Lab2/Q2a.pdf", bbox_inches="tight")

# ================================= b ==============================================

fig, ax = plt.subplots()
ax.plot(N, gauss_err, color="hotpink")

ax.set_xlabel("N")
ax.set_ylabel("error estimate $|I_{2N}-I_N|$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("../Lab2/Q2b.pdf", bbox_inches="tight")

# ================================= c ==============================================

fig, ax = plt.subplots()
ax.plot(N, gauss_err_rel, color="hotpink")

ax.set_xlabel("N")
ax.set_ylabel("relative error")
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("../Lab2/Q2c.pdf", bbox_inches="tight")
