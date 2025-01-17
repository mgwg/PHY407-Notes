import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl

# change default matplotlib settings
# increase font size
# change default colors
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=["hotpink", "cornflowerblue", "yellowgreen"]) 

# data
x = np.array([-38.04, -35.28, -25.58, -28.80, -30.06])
y = np.array([27.71, 17.27, 30.68, 31.50, 10.53])

# ==================================== (a) ====================================
# write code to fit dataset for constants of best fit ellipse

def get_y(x, p):
    """returns y values of ellipse parameterized by a for a given array of xs

    Args:
        p (np.ndarray): array containing ellipse parameters (A, B, C, D, F, G)
        x (np.ndarray)
    """
    A, B, C, D, F, G = p

    # discriminant of ellipse equation
    # we only care about xs where the discriminant is real
    d = (2*F+2*B*x)**2 - 4*C*(G+2*D*x+A*x**2) 

    # now, given an array of x, we want to keep only the values where y is real
    if len(x) > 1:
        mask = np.where(d < 0)
        # set values where d < 0 to NaN
        d[mask] = np.NaN

        # y values corresponding to plus and minus roots of equation
        ym = (-2*F-2*B*x - np.sqrt(d))/(2*C)
        yp = (-2*F-2*B*x + np.sqrt(d))/(2*C)

    # treat separately if x is a float
    elif d < 0:
        ym, yp = None, None

    return ym, yp

def X(x, y):
    """ X = (x^2, xy, y^2, x, y, 1) such that f(a, X) = X·a = 0, where 
    a = (A, 2B, C, 2D, 2F, G).
    If x, y are arrays, X must be a matrix where each row corresponds to an 
    element in x, y

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates

    Returns:
        np.ndarray: the 2D array X = (x^2, xy, y^2, x, y, 1)
    """
    N = len(x) # number of data points
    X_mtx = np.zeros((N, 6)) # initialize matrix

    for i in range(N):
        X_mtx[i] = (x[i]**2, x[i]*y[i], y[i]**2, x[i], y[i], 1)

    return X_mtx

def S(x, y):
    """returns S=X^T·X

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
    """
    X_mtx = X(x, y)
    S = np.dot(X_mtx.T, X_mtx)

    return S

def Y():
    """returns the constraint matrix Y, where a^T Y a = 4 AC - B^2
    """

    # initialize matrix
    Y_mtx = np.zeros((6, 6))
    # set constraint elements
    Y_mtx[1, 1] = 1/4
    Y_mtx[0,2] = -4

    return Y_mtx

def fit_ellipse(x, y):
    """returns fit parameters A,B,C,D,F,G of ellipse for some input arrays x, y
    following the equation Ax^2 + Bxy + Cy^2 + Dx + Fy + G = 0
    """

    # construct eigenvalue problem
    S_inv_mtx = np.linalg.inv(S(x, y))
    Y_mtx = Y()
    E, V = np.linalg.eig(np.dot(S_inv_mtx, Y_mtx))
    # take largest eigenvalue and corresponding eigenvector as solution
    i = np.argmax(np.abs(E))
    a = V[:,i]

    # find parameters from the vector a = [A, 2B, C, 2D, 2F, G]
    p = np.copy(a)
    p[[1, 3, 4]] /= 2

    return p

p = fit_ellipse(x, y)
xs = np.linspace(-40, 5, 5000)
ys = get_y(xs, p)

print("Best fit parameters:", np.real(p))
# ==================================== (b) ====================================

fig, axs = plt.subplots(2,1, figsize=(8, 8), sharex=False, 
                        height_ratios = [3,1])
axs[0].plot(xs, ys[0], color="plum", linewidth=2, label="fit")
axs[0].plot(xs, ys[1], color="plum", linewidth=2)
axs[0].plot(x, y, markersize=10, marker='*', linestyle="", 
         color="hotpink", label="data")
axs[0].plot(0, 0, color="gold", marker='$\U00002600$', linestyle="", 
            markersize=20, label="Sun")

axs[0].set_xlabel("x [AU]")
axs[0].set_ylabel("y [AU]")
axs[0].legend(loc=0)
fig.tight_layout()

# get residuals
y_fit = get_y(x, p)
# first we calculate the difference between each point and 
# both the + and - root
diff =  y_fit - y
# take the residual as the minimum difference  
# (i.e. either the + or the - root)
i = np.argmin(np.abs(diff), axis=0) # array of indices corresponding to 
                                    # either root
# get the difference at those indices
# this is a bit convoluted because we want to keep the sign of the difference
residuals = [diff[i[j], j] for j in range(len(x))]
                                    
axs[1].plot(x, residuals/y*100, linestyle="", marker='*', markersize=10)
axs[1].set_ylabel("normalized\nresiduals (%)")
axs[1].axhline(0, linestyle="--", color="lightgrey")

plt.savefig("Q1a.pdf", bbox_inches="tight")
