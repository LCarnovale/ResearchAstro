import numpy as np
import fit_Bezier as fb
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

def get_curvature(track, X, Y):
    """ Calculates the curvature in the given `X` and `Y` axes (must be
    axis labels accessible with `track[X]` etc.) and returns 3 arrays:
        `*inv_radius*`: Inverse of the radius of curvature. Ranges from
            0 (very straight) to infinity (more curvy). The actual radius
            is simply 1 / inv_radius.
        `*tangent*`: The tangent vector at all points along the curve. The
            length of these vectors depends on the distance between the points,
            and is mostly meaningless.
        `*dists*`: The cumulative distance along the curve at each point.
    """

    X_ax = track[X]
    Y_ax = track[Y]

    points = np.array([X_ax, Y_ax]).T
    dists = fb.get_dist_array(points)
    # Get tangent and curvature arrays:
    dX = np.gradient(X_ax, dists)
    dY = np.gradient(Y_ax, dists)

    d2X = np.gradient(dX, dists)
    d2Y = np.gradient(dY, dists)

    tangent = np.array([dX, dY]).T
    curvature = np.array([d2X, d2Y]).T
    inv_radius = np.linalg.norm(curvature, 2, 1)

    return (inv_radius, tangent, dists)

def set_curvature_dist(track):
    """
    Create and return columns for:
        curv_inv_rad, curv_norm, tangent, path_len
    Must be able to see 'Teff' and 'L'
    """
    T_ax = track['Teff']
    L_ax = track['L']
    points = np.array([T_ax, L_ax]).T
    dists = fb.get_dist_array(points)
    # Get tangent and curvature arrays:
    dT = np.gradient(T_ax, dists)
    dL = np.gradient(L_ax, dists)

    d2T = np.gradient(dT, dists)
    d2L = np.gradient(dL, dists)

    tangent = np.array([dT, dL]).T
    curvature = np.array([d2T, d2L]).T
    inv_radius = np.linalg.norm(curvature, 2, 1)

    return (inv_radius, inv_radius/max(inv_radius), tangent, dists)

    # # Get points of minimum curvature
    # track['curv_inv_rad'] = inv_radius
    # track['curv_norm'] = inv_radius / max(inv_radius)
    # track['tangent'] = tangent
    # track['path_len'] = dists
