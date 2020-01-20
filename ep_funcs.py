import numpy as np
import matplotlib.pyplot as plt

def zams_ep(track):
    window_width = 100
    # curv_max = np.argmax(track.curv_inv_rad)
    min_T = np.argmin(track.Teff[0: window_width])
    # min_T += curv_max - window_width
    return min_T

def h_depletion(track):
    try:
        track.center_h1
    except:
        raise Exception("Unable to see center_h1 field in track.")
    else:
        mask = track.center_h1 < 1e-9
        zero = np.flatnonzero(mask)[0]
        return zero

# def density_eps(track):
#     d_ax = np.log10(track.M) - 3*track.log_R
#     # abs_d = abs(density + 2)
#     y_ax = np.arange(len(track))
#     idx = np.interp(2., -d_ax, y_ax)
#     return idx

def delta_nu(track):
    L = len(track.delta_nu[track.delta_nu > 10])
    L -= 1
    d_ax = np.arange(len(track))
    idx = np.interp(10, track.delta_nu[[L+1, L]], [L+1, L])
    return idx
    # # plt.plot(d_ax, track.delta_nu[L//2:])
    # # plt.show()
    # idx = np.interp(-10, d_ax, -track.delta_nu[300:]) + 300
    # return idx
