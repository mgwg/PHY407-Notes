import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# change default matplotlib settings
# increase font size
# change default colors
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=["hotpink", "cornflowerblue", "yellowgreen"]) 

# a
path = "sp500.csv"
close = np.loadtxt("sp500.csv", skiprows=1, usecols=(4), delimiter=",")

fig, axs = plt.subplots(2, 1, figsize=(8, 8), height_ratios=(1,1))
axs[0].plot(close)
axs[0].set_xlabel("Business day")
axs[0].set_ylabel("Closing value")

# b
fft_coeffs = np.fft.rfft(close)
ifft_vals = np.fft.irfft(fft_coeffs)
close_trunc = close[1:]

axs[0].plot(ifft_vals, color="plum")
axs[1].plot((close_trunc-ifft_vals)/close_trunc)
axs[1].set_ylabel("Relative difference (%)")
axs[1].set_xlabel("Business day")


# c
thresh = 1000
sort_ind = np.argsort(fft_coeffs)
fft_coeffs_filtered = np.copy(fft_coeffs)
fft_coeffs_filtered[np.abs(fft_coeffs) > thresh] = 0
axs[0].plot(fft_coeffs_filtered)


plt.tight_layout()