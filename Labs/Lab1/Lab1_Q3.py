import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

# colors used for plotting
colors = ["hotpink", "cornflowerblue"]

# increase defualt font size
matplotlib.rcParams.update({'font.size': 15})

def p(u):
    return (1-u)**8

def q(u):
    """Taylor expansion of (1-u)^8 to 8th degree
    """
    return 1- 8*u + 28*u**2 - 56*u**3 + 70*u**4 -56*u**5 + 28*u**6 - 8*u**7 + u**8

# ===================================== a =====================================

# Define array of u's from 0.98 to 1.02
u_array = np.linspace(0.98, 1.02, 500)
# calculate p and q at each u
p_array = p(u_array)
q_array = q(u_array)

# plot the result
fig, ax = plt.subplots()
ax.plot(u_array, p_array, color=colors[0], label="p(u)")
ax.plot(u_array, q_array, color=colors[1], label="q(u)")
# add labels and legend
ax.legend()
ax.set_xlabel("u")

# save figure
plt.savefig("3a.pdf")

# ===================================== b =====================================
# compute difference between p(u) and q(u) for u from 0.98 to 1.02
difference = p_array - q_array

# make 20 histogram bins of the difference
cts, bin_edges = np.histogram(difference, 20)
# calcualte the bin centres from the bin edges
bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
# calcualte an appropriate width for each histogram bar
bar_width = (bin_edges[1]-bin_edges[0])/2

# plot the difference on top and histogram on the bottom
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(u_array, difference, color=colors[0])
ax[1].bar(bin_centres, cts, width=bar_width, color=colors[0])
# set plot labels
ax[0].set_xlabel("u")
ax[0].set_ylabel("p(u)-q(u)")
ax[0].text(0.971, 2.5e-14, "a)")
ax[1].set_xlabel("p(u)-q(u)")
ax[1].set_ylabel("counts")
ax[1].text(-3.3e-14, 65, "b)")
# save figure
plt.savefig("3b.pdf", bbox_inches="tight")

# calculate and print the standard deviation, and estimated error
print(f"3b: std of p(u)-q(u) {np.std(difference)}")
print(f"3b: estimate of error {31e-16 * 500**0.5 * (np.mean(np.abs(difference))**2)**0.5}")

# ===================================== c =====================================
# initialize and set u, p, q arrays as before
u_array = np.linspace(0.978, 0.985, 500)
p_array = p(u_array)
q_array = q(u_array)
# calculate the fractional error
fractional_error = np.abs(p_array - q_array)/np.abs(p_array)

# plot
fig, ax = plt.subplots()
ax.axhline(1, color="lightgray", linestyle="--", label="100% error")
ax.plot(u_array, fractional_error, color=colors[0])
# set legend and labels
ax.legend()
ax.set_ylabel("$\\frac{|p(u)-q(u)|}{|p(u)|}$")
ax.set_xlabel("u")
# save figure
plt.savefig("3c.pdf", bbox_inches="tight")

# ===================================== d =====================================
# initialize array of us
u_array = np.linspace(0.98, 1.02, 500)
# calculate f as in assignment description
f = u_array**8/((u_array**4)*(u_array**4))
# plot and save plot
fig, ax = plt.subplots()
ax.plot(u_array, f-1, color=colors[0])
ax.set_xlabel("u")
ax.set_ylabel("$u^8/(u^4 u^4) - 1$")
plt.savefig("3d.pdf", bbox_inches="tight")
# print std of f
print(f"3d: std: {np.std(f)}")