import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# increase defualt font size
matplotlib.rcParams.update({'font.size': 15})

# Gaussian quadrature functions from Q1
def gaussxw(N):
    """From week 3 practical. Returns array of points x and weights w for gaussian 
    quadrature integration, for integration range from -1 to 1

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
    """From week 3 practical. Returns array of points x and weights w for gaussian 
    quadrature integration, for integration range from a to b

    Args:
        N (int): number of integration slices
        a (float): start of integration interval
        b (float): end of integration interval
    """
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gaussian_quad(N, a, b, f, args=[]):
    """returns integral of f computed using gaussian quadrature, using N points 
    from  a to b

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

# ================================= a ==============================================
def H(n, x):
    """returns hermite polynomial of degree n, evaluated at x
    """
    if n > 1:
        return 2*x*H(n-1, x) - 2*(n-1)*H(n-2, x)
    elif n == 1:
        return 2*x
    elif n == 0:
        return 1
# ================================= b ==============================================

def psi(n, x):
    """ returns the wavefunction of the nth energy level of a 1D quantum harmonic 
    oscillator evaluated at position x
    """
    coeff = 1/np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
    return coeff*np.exp(-x**2/2)*H(n, x)

# initialize values of n and x to evaluate over
ns = [0, 1, 2, 3]
xs = np.linspace(-4, 4, 100)
# plotting colors
colors = ["hotpink", "cornflowerblue", "yellowgreen", "mediumpurple"]

# plot
fig, axs = plt.subplots()

# evaluate and plot the wave function for each n
for i in range(len(ns)):
    n = ns[i]
    psi_arr = psi(n, xs)
    axs.plot(xs, psi_arr, color=colors[i], label=f"n={n}")

axs.set_xlabel("x")
axs.set_ylabel(f"$\psi_n(x)$")
axs.legend()
plt.savefig("../Lab2/Q3b.pdf", bbox_inches="tight")

# ================================= c ==============================================

def rms_integrand(z, n):
    """the integrand of the rms position squared, normalized to integrate the 
    improper integral

    Args:
        z (float): integration parameter
        n (int): energy level

    Returns:
        float
    """
    x = np.tan(z)
    return x**2 * psi(n, x)**2 / np.cos(z)**2

def potential_energy(n):
    """the potential energy of a quantum harmonic oscillator with energy level n, 
    calculated using gaussian quadrature with 100 points

    Args:
        n (int)

    Returns:
        float
    """
    N = 100
    # calculate the potential energy as the rms squared / 2
    return gaussian_quad(N, -np.pi/2, np.pi/2, rms_integrand, [n])/2

# print the potential energy for n from 0 to 10
for n in range(11):
    print(n, potential_energy(n))