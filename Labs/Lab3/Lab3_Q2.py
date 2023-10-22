import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.constants import c, h, k # in m/s, J/Hz, J/K

# change default matplotlib settings:
#   - increase font size
#   - change default colors

colors = color=["hotpink", "cornflowerblue", "yellowgreen"]

mpl.rcParams.update({'font.size': 12})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

# ========================== integration functions ============================
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

# ===================================== a) ====================================

def I(l, t):
    """Returns power radiated by perfect blackbody at wavelength l and 
    temperature t

    Args:
        l (float, array): wavelength of interest in m
        t (float, array): temperature of blackbody in K

    Returns:
        (float, array)
    """
    return 2*np.pi*h*(c**2)/(l**5 * (np.exp(h*c/(l*k*t))-1))

def E_total(T):
    """Returns analytical value total energy emitted by perfect blackbody at 
    temperture T across all wavelengths

    Args:
        T (float, array): temperature of blackbody in K

    Returns:
        (float, array)
    """
    return 2*np.pi*h*(c**2) * k**4 * np.pi**4 * T**4 / (15 * c**4 * h**4)

def E_total_int(t, N):
    """Returns total energy emitted by perfect blackbody at temperture T across 
    all wavelengths calculated numerically using Gaussian Quadrature

    Args:
        t (float, array): temperature of blackbody in K
        N (int): number of integration points

    Returns:
        (float, array)
    """
    # calculate positions and weights for Gaussian Quadrature. 
    # Since the total energy is integrated from 0 to infinity, but we will 
    # apply a change of variables  l = z/(1-z) to the integrand to evaluate 
    # it numerically from 0 to 1
    z, w = gaussxwab(N, 0, 1)
    # get 2D arrays of Zs and Ts, where the x axis is transformed wavelength
    # and the y axis is temperature
    Z, T = np.meshgrid(z, t)
    # integrand after change of variables
    integrand = I(Z/(1-Z), T)/(1-Z)**2
    # calculate the integral for each temperature in t 
    # the sum over axis 1 sums over all wavelengths
    return np.sum(w*integrand, axis=1)

def eta(t, l, w):
    """Returns the efficiency of a light bulb modelled as a perfect blackbody
    with temperture T across

    Args:
        t (float, array): temperature of blackbody in K
        l (array): wavelengths to evaluate integrand at
        w (array): weights for Gaussian Quadrature corresponding to each point
                    in l
    Returns:
        (float, array)
    """
    # get 2D arrays of Ls and Ts, where the x axis is wavelength, and the
    # y axis is temperature
    L, T = np.meshgrid(l, t)
    # calculate energy from wavelengths in the range of l, for each temperature
    # in t. The sum over axis 1 sums over all wavelengths for each t
    E12 = np.sum(w*I(L, T), axis=1)
    # calculate total energy using the analytical expression
    Etot = E_total(t)

    return E12/Etot 

def get_etas(T, l1=380e-9, l2=780e-9, lN=10000):
    """Returns the efficiency of a perfect blackbody for temperature T, 
    for radiation between 380 nm and 780 nm. 
    Energy in the wavelength range of interest is calculated numerically
    using Gaussian Quadrature

    Args:
        Tmin (float, optional): Temperature in K.
        l1 (float, optional): Lower bound of wavelength, in m. 
                                Defaults to 380 nm.
        l2 (float, optional): Upper bound of wavelength in m. 
                                Defaults to 780 nm.
        lN (int, optional): Number of integration points. Defaults to 10000.

    Returns:
        (array, array): array of temperatures in K, 
                        corresponding array of efficiencies
    """

    l, w = gaussxwab(lN, l1, l2)
    T = np.linspace(Tmin, Tmax, Tnvals)
    etas = eta(T, l, w)

    return etas

# constants given in the lab manual
Tmin, Tmax = 300, 10000 # in K
Tnvals = 100 # number of points over T
T = np.linspace(Tmin, Tmax, Tnvals)

# First, we want to figure out how many integration points to use

# array of num integration points
Ns = [5000, 8000, 10000]
# scale factor to change units in plot
scale = 1e-6 

fig, axs = plt.subplots(2, 1, height_ratios=(1,1), sharex=True)
# calculate total energy using the analytical expression
# use this as the reference "real" value
E_tot_real = E_total(T)

for i in range(len(Ns)):
    N = Ns[i]
    color = colors[i]
    # calculate total energy numerically with N integration points
    E_tot = E_total_int(T, N)

    # plot
    axs[0].plot(T,  E_tot*scale, label = f"N={N}", color=color)
    if i >= 1: # don't plot error for N = 5000, since it's too large
        axs[1].plot(T, (E_tot-E_tot_real)/E_tot_real*100, label = f"N={N}", 
                    color=color)

axs[0].plot(T, E_tot_real*scale, linestyle = ":", label="Analytic value", color="dimgrey")
axs[0].legend(loc=0)
axs[1].axhline(0, linestyle="--", color="lightgray")
axs[0].set_ylabel(f"$E(0,\infty)$ [W/um]")
axs[1].set_ylabel("Fractional error [%]")
axs[1].set_xlabel("T [k]")

plt.savefig("Q2a.pdf", bbox_inches="tight")
# ===================================== b) ====================================
lN = 10000
l1_vis, l2_vis = 380e-9, 780e-9 # in m
etas_vis = get_etas(T, l1_vis, l2_vis, lN) # use lN=10000 based on the plot above

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(T, etas_vis)
ax[0].set_xlabel("T [K]")
ax[0].set_ylabel(f"$\eta$ visible")

# plot is saved in next section
# ===================================== c) ====================================

def golden_ratio(x1, x4, f, args=[]):
    """Calculates temperature of max efficiency for a perfect blackbody
    using golden ratio

    Args:
        x1 (float): Starting lower bound of interval
        x4 (float): Starting upper bound of interval
        f (function): function to be maximized
        args (list, optional): additional parameters to pass to f. 
                    Defaults to empty list.

    Returns:
        float, float: xmax, f(xmax)
    """
    # golden ratio
    z = (1+np.sqrt(5))/2

    while np.abs(x1 - x4) >= 1: # accuracy to within 1K
        # get x2, x3
        x2 = -(x4-x1)/z + x4
        x3 = (x4-x1)/z + x1
        # calculate function at x2, x3
        y2 = f(x2, *args)
        y3 = f(x3, *args)
        # update bounds such that the maximum is always enclosed in 
        # the new bounds
        if y2 > y3:
            x4 = x3
        else:
            x1 = x2 

    xmax = np.mean([x1, x4])
    ymax = f(xmax, *args)

    return xmax, ymax

# T1, T4 starting guesses based on plot in b)
T1 = 6000 # K
T4 = 8000 # K

# the integration occurs over the same interval, so save the position and 
# weights for Gaussian Quadrature to reduce computation time
l, w = gaussxwab(lN, l1_vis, l2_vis)

Tmax_vis, eta_max_vis = golden_ratio(T1, T4, eta, [l, w])
print("Maximum visible temperature and efficiency:", Tmax_vis, eta_max_vis)
# add max point to plot
ax[0].axvline(Tmax_vis, linestyle="--", color="lightgray")
ax[0].plot(Tmax_vis, eta_max_vis, marker="*", markersize=15, color="plum", 
        label="maximum efficiency", linestyle="")
ax[0].legend(loc=0)
ax[0].annotate('a)', xy=(-2000, eta_max_vis*1.1), annotation_clip=False)
# ===================================== d) ====================================

# update parameters
l1_IR, l2_IR = 780e-9, 2250e-9 # in m
etas = get_etas(T, l1_IR, l2_IR, lN)

ax[1].plot(T, etas)
ax[1].set_xlabel("T [K]")
ax[1].set_ylabel(f"$\eta$ IR")

# repeat calculation of maximum from part c)

T1 = 2000 # K
T4 = 4000 # K
l, w = gaussxwab(lN, l1_IR, l2_IR)
Tmax_IR, eta_max_IR = golden_ratio(T1, T4, eta, [l, w])

print("Maximum visible temperature and efficiency:", Tmax_IR, eta_max_IR)

# add max point to plot
ax[1].axvline(Tmax_IR, linestyle="--", color="lightgray")
ax[1].plot(Tmax_IR, eta_max_IR, marker="*", markersize=15, color="plum", 
        label="maximum efficiency", linestyle="")
ax[1].legend()

ax[1].annotate('b)', xy=(-2000, eta_max_IR*1.1), annotation_clip=False)

plt.savefig("Q2.pdf", bbox_inches="tight")