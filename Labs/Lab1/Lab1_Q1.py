import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import jv # Bessel functions
import matplotlib 

# colors used for plotting later
colors_light = ["plum", "skyblue", "yellowgreen"]
colors_dark = ["deeppink", "royalblue", "lawngreen"]
colors = ["hotpink", "cornflowerblue"]

# increase defualt font size
matplotlib.rcParams.update({'font.size': 15})

# ===================================== a =====================================
def simpson(N, a, b, f, args=[]):
    """
    Uses Simpson's rule to compute the integral of f from a to b, with N points
    Args:
        N (int)
        a (float)
        b (float)
        f (function)
        args (list): list of optional parameters
    
    Returns:
        float: integral of f over [a, b], calculated using Simpson's rule
    """
    # step size
    h = (b-a)/N 

    # compute sum of f over all points
    # start with f(a) and f(b)
    sum_f = f(a, *args) + f(b, *args)
    # iterate through odd and even k
    for k in range(1, N):
        if k % 2: # k is odd
            sum_f += 4*f(a + k*h, *args)
        else: # k is even
            sum_f += 2*f(a + k*h, *args)

    return 1/3*h*sum_f

def J_integrand(theta, m, x):
    """Integrand of the Bessel function J_m(x), excluding constant factors
    """
    return np.cos(m*theta - x*np.sin(theta))

def J(m, x):
    """ Value of Bessel function J_m(x), calculated using Simpson's rule with 
        N = 1000
    """
    return 1/np.pi * simpson(1000, 0, np.pi, J_integrand, args=[m, x])

# Initialize figure and axes
fig, ax = plt.subplots()

# Compute and plot the mth Bessel function from x = 0 to x = 20
x_arr = np.linspace(0, 20, 100)
for m in range(3):
    J_simpson = J(m, x_arr)
    ax.plot(x_arr, J_simpson, color = colors_light[m], label = f"m={m}")

# Add labels to plot
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("$J_m(x)$")
# ax.set_title(f"Bessel functions $J_m(x)$")
ax.axhline(0, color="lightgray")
ax.grid(color="gainsboro")
# save the plot
plt.savefig("1a.pdf", bbox_inches="tight")

# ===================================== b =====================================

# Top axis plots Bessel function
# Bottom plots the difference between Simpson's rule and Scipy
fig, ax = plt.subplots(2, 1)

x_arr = np.linspace(0, 20, 100)
for m in range(3):
    # As in a), compute the mth Bessel function from x = 0 to x = 20
    J_simpson = J(m, x_arr)
    J_scipy = jv(m, x_arr)

    # Plot each calculation and the difference
    ax[0].plot(x_arr, J_simpson, color = colors_light[m], label = f"m={m}")
    ax[0].plot(x_arr, J_scipy, color = colors_dark[m], 
               linestyle=(0,(1,3)))
    
    ax[1].plot(x_arr, np.abs(J_simpson - J_scipy), color = colors_light[m])

# Add linestyles differentiating Scipy and Simpson's rule lines to legend
handles, labels = ax[0].get_legend_handles_labels()
line_solid = Line2D([0], [0], label="Simpson's rule", color='darkgray')
line_dash = Line2D([0], [0], linestyle=(0, (1,3)), label='Scipy', 
                   color='darkgray')
handles.extend([line_solid, line_dash])

# Add plot labels and legend
ax[0].legend(handles = handles, loc="center left", bbox_to_anchor=(1, 0))
ax[0].set_ylabel("$J_m(x)$")
ax[0].axhline(0, color="lightgray")
ax[0].grid(color="gainsboro")

ax[1].set_xlabel("x")
ax[1].set_yscale("log")
ax[1].set_ylabel("|Simpson - Scipy|")
ax[1].axhline(0, color="lightgray")
ax[1].grid(color="gainsboro")

# save figure
plt.savefig("1b.pdf", bbox_inches="tight")

# ===================================== c ===================================== 
def I(r, wavelength):
    """ Intensity of diffracted light from point light source using
        Simpson's rule

    Args:
        r (float): distance from beam centre in um
        wavelength (float): wavelength in um

    Returns:
        float
    """
    k = 2*np.pi/wavelength # wavenumber
    return (J(1, k*r)/(k*r))**2 

R = 1000 # Define area of interest from beam centre in nm
num_grid_points = 100
wavelength = 500

# Initialize arrays of x, y as cartesian coordinates over R
x = np.linspace(-R, R, num_grid_points)
y = np.linspace(-R, R, num_grid_points)
# Compute matrices representing the x/y coordinates at each grid point
x_matrix, y_matrix = np.meshgrid(x, y)
# Compute the radius at each point
r_matrix = np.sqrt(x_matrix**2+y_matrix**2)
# Compute the intensity at each point
intensity = I(r_matrix, wavelength)

# Plot intensity
fig, ax = plt.subplots()
plot = ax.imshow(intensity, vmax=0.01, cmap ="RdPu_r", extent=[-1000, 1000, -1000, 1000])
# Show color bar and set plot labels
cbar = fig.colorbar(plot, ax=ax)
cbar.ax.set_ylabel("Intensity")
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title("Intensity of diffracted light")

# Save figure
plt.savefig("1c.pdf")
