import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# increase defualt font size
matplotlib.rcParams.update({'font.size': 15})

def f(x):
    return np.exp(2*x)

def dmf(x, m):
    """mth derivative of f evaluated at x
    """
    return (2**m)*np.exp(2*x)

# ================================= a ==============================================

def central_difference(h, x, f):
    """f'(x) calculated using central differences with step size h
    """
    return (f(x+h/2) - f(x-h/2))/h

# initialize array of step sizes
hvals = np.logspace(-16, 0, 17)
# calculate and print corresponding f'(0)
dfdx = central_difference(hvals, 0, f)
print("1 a) errors: ", dfdx)

# ================================= b ==============================================

# calculate errors of central difference
errors = np.abs(dfdx - 2)

# define machine error and x value of interest
C = 1e-16 
x = 0

# calculate the expected error
roundoff_err = 2*C*f(x)/hvals
approx_err = hvals**2*dmf(x, 3)/24
expected_err = roundoff_err + approx_err

# get estimate of the optimal step size and corresponding error
h_opt = (24*C*f(x)/dmf(x, 3))**(1/3)
eps_min = ((9/8)*C**2*f(x)**2*dmf(x, 3))**(1/3)

# also get index for the minimum of the calculated errors
imin = np.argmin(errors)

# plot error vs h, expected error vs h, and expected minimum
plt.plot(hvals, expected_err, color="lightgray", label="expected error", 
         linestyle="--")
plt.plot(hvals, errors, color="hotpink")
plt.xlabel(f"Step size $h$")
plt.ylabel(f"Error $|f'(0)-2|$")
plt.xscale("log")
plt.yscale("log")

plt.plot(h_opt, eps_min, marker="*", markersize=15, color="plum", 
         label="estimated minimum", linestyle="")
plt.legend()
plt.savefig("../Lab2/Q1b.pdf", bbox_inches="tight")

print(f"1 b): estimated step size {h_opt}, error {eps_min}")
print(f"1 b): actual step size {hvals[imin]}, error {errors[imin]}")
# ================================= c ==============================================


def delta(f, x, m, h):
    """ from the lab manual: mth derivative of f, evaluated at x using repeated 
    central differences with step size h
    """
    if m > 1:
        return (delta(f, x + h/2, m - 1, h) - delta(f, x - h/2, m-1, h))/(h)
    else:
        return (f(x + h/2) - f(x - h/2))/(h)

# initialize x
x = 0
# get optimal value of h
h_opt = (24*C*f(x)/dmf(x, 3))**(1/3)
# calculate and print list of derivatives of f at x=0
dfs = [delta(f, x, m, h_opt) for m in range(1, 11)]
print("1 c) derivatives: ", dfs)