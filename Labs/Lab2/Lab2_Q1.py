import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(2*x)

def d3f(x):
    return 8*np.exp(2*x)

# ================================= a ==============================================

def central_difference(h, x, f):
    return (f(x+h/2) - f(x-h/2))/h

hvals = np.logspace(-16, 0, 17)
dfdx = central_difference(hvals, 0, f)
print(dfdx)

# ================================= b ==============================================

errors = np.abs(dfdx - 2)
C = 1e-16 
x = 0

roundoff_err = 2*C*f(x)/hvals
approx_err = hvals**2*d3f(x)/24
expected_err = roundoff_err + approx_err

h_opt = (24*C*f(x)/d3f(x))**(1/3)
eps_min = ((9/8)*C**2*f(x)**2*d3f(x))**(1/3)

# plt.plot(hvals, roundoff_err, linestyle="--")
# plt.plot(hvals, approx_err, linestyle="--")

plt.plot(hvals, errors)
plt.xlabel("Step size")
plt.ylabel("Error")
plt.xscale("log")
plt.yscale("log")

plt.plot(hvals, expected_err)
plt.plot(h_opt, eps_min, marker="*", markersize=20)

# ================================= b ==============================================


def delta(f, x, m, h):
    if m > 1:
        return (delta(f, x + h/2, m - 1, h) - delta(f, x - h/2, m-1, h))/(h)
    else:
        return (f(x + h/2) - f(x - h/2))/(h)

dfs = [delta(f, 0, m, h_opt) for m in range(1, 11)]
print(dfs)