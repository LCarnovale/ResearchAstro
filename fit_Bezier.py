# Author: Leo Carnovale (leo.carnovale@gmail.com)

"""
Functions to fit an arbitrary order Bezier curve to an arbitrary path.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb


def A(row, col, size):
    """Get the value of the element in a 
    lower triangular coeffecient matrix
    for a bezier curve of order `size-1`, ie
    for a matrix of size `size`."""
    out = ( 
        (-1)**(abs(row - col)) * (
            comb(row, col) * comb(size - 1, row)
        )
    )
    try:
        out[row < col] = 0
    except:
        if row < col: out = 0

    return out

_generated = {}
def generate_matrix(N):
    """ Generate a coeffecient matrix of size `N` for the calculation
    of Bezier curves. This returns `M` in `B(t) = T(t) * M * P`."""
    global _generated
    if N in _generated:
        return _generated[N]
    else:
        rows = np.array([[row for col in range(N)] for row in range(N)])
        cols = np.array([[col for col in range(N)] for row in range(N)])
        M = A(rows, cols, N)
        _generated[N] = M
        return M
    
def T(t, n):
    """ Return an array of `[1, t^1, t^2, ..., t^(n-1)]`
    for every value of `t`. `t` can be scalar or a 1-D array.
    """
    a = np.array([np.asarray(t)**i for i in range(n)]).T
    size = np.size(t)
    if size > 1:
        return a.reshape(size, -1, 1)
    else:
        return a.reshape(-1, 1)

def interp_path(path, distance, normalised=False):
    """ Interpolate along a given path to find the location of
    a point at any given distance.
    `path` should be an array of coordinates,
    `distance` should be a float value for a distance shorter than 
    the length of the path, and greater than zero.

    `normalised`: If true, then scale the input distance so that 1.0
    corresponds to the end of the path.

    Return an array of points for each value in distance.
    """ 

    distance = np.asarray(distance)

    # Get a distance array:
    diffs = path[1:] - path[:-1]         # vector differences
    
    dists = np.linalg.norm(diffs, 2, 1)  # distance from point i to i+1
    s = np.array([0, *dists.cumsum()])   # Distances of points along the path

    if normalised: s /= s[-1]
    
    if (np.any(distance > s[-1])):
        raise ValueError(f"Distance must be within the length of the path. \nPath length / given distance: {s[-1]} / {distance}")
    # map x and y values to distance along the line.
    # interpolate for the x and y value
    x_interp = np.interp(distance, s, path[:, 0])
    y_interp = np.interp(distance, s, path[:, 1])

    return np.array([x_interp, y_interp]).T

def curve_eval(t, control_points):
    """ `control_points` should be a N x 2 array.
    `t` should be an array like of floats between 0 and 1. 
    Returns a list of points corresponding to the given t values."""
    t = np.asarray(t).reshape(-1, 1)
    N = len(control_points)
    M = generate_matrix(N)
    Tm = T(t, N)

    points = (Tm*M).dot(control_points)
    points = points.sum(axis=1)
    return points

def get_curve_dist_func(control_points, n_points=100):
    """ Returns a function to approximate the distance (by interpolation)
    of a point on the curve with a `t` value. ie, returns a function of the form:

        f = get_curve_dist_func(points)
        f(0.5) -> distance along curve for t = 0.5.
    
    `f(1.0)` will give the full length of the curve.

    `n_points`: default 100, numper of divisions of bezier curve to use
    for interpolation.
    """
    t_ax = np.linspace(0, 1, n_points)
    curve_points = curve_eval(t_ax, control_points)

    # Get distances
    diffs = curve_points[1:] - curve_points[:-1]
    dists = np.linalg.norm(diffs, 2, 1)
    s = np.array([0, *dists.cumsum()])

    f = lambda t : np.interp(t, t_ax, s)

    return f

def get_inverse_dist_func(control_points, n_points=100):
    """ Returns a function to convert a normalised distance
    into a `t` value. Essentially the inverse of the function
    returned by `get_curve_dist_func`, except that both the 
    domain and range of the returned function is [0, 1].
    """
    t_ax = np.linspace(0, 1, n_points)
    curve_points = curve_eval(t_ax, control_points)

    diffs = curve_points[1:] - curve_points[:-1]
    dists = np.linalg.norm(diffs, 2, 1)
    s = np.array([0, *dists.cumsum()])

    f = lambda t : np.interp(t, s, t_ax)

    return f


def get_dist_path_curve(t, path, control_points, n_points=100):
    """ Get the distance between corresponding points on
    a Bezier curve built from the give `control_points`,
    and on the given `path`.

    `n_points`: number of points to use to interpolate distance
    along Bezier curve.

    Returns an array of floats the same length as `t`.
    """
    # Get distances along curve:
    dist_f = get_curve_dist_func(control_points, n_points)
    dists  = dist_f(t) # -> distances of points along the bezier curve.
    bez_points = curve_eval(t, control_points)

    # Get points along path:
    norm_dists = dists / dist_f(1.)
    path_points = interp_path(path, norm_dists, normalised=True)

    # Get differences between points
    diffs = bez_points - path_points
    dists = np.linalg.norm(diffs, 2, 1)
    return dists

def fit_curve(path, control_points, n_points=100):
    """ Create a function that can be fit to with `scipy`'s `optimize.curve_fit`
    more easily than `get_dist_path_curve`, and then attempt to fit the curve,
    and give the optimized control points.
    
    Returns the optimized control points and the 
    flattened parameters' covariance matrix."""

    con_shape = control_points.shape
    # Create a function to take in the flattened control points.
    def f(t, *args):
        con_points = np.asarray(args).reshape(con_shape)
        dists = get_dist_path_curve(t, path, con_points, n_points)
        return dists

    flat_c = control_points.flatten()

    upper_bound = flat_c.copy()
    upper_bound[2:-2] = np.inf
    lower_bound = flat_c - 1e-6
    lower_bound[2:-2] = -np.inf

    popt, pcov = curve_fit(f,
        np.linspace(0, 1, n_points),
        np.zeros(n_points), 
        p0=flat_c,
        bounds=(lower_bound, upper_bound)
    )

    # popt will have the flattened optimized control points.
    return popt.reshape(con_shape), pcov



def plot_curve(control_points, ax=None, n_points=100):
    if ax == None: ax = plt.gca()
    t_ax = np.linspace(0, 1, n_points)
    points = curve_eval(t_ax, control_points)
    return ax.plot(*points.T)

if __name__ == "__main__":
    path = np.array([np.linspace(1, 3, 100), np.linspace(1, 3, 100)**2]).T
    conpoints = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])

    fig = plt.figure()
    ax1, ax2 = fig.subplots(ncols=2)

    ax1.plot(*path.T)
    ax1.plot(*conpoints.T)
    plot_curve(conpoints, ax1)

    conpoints_opt = fit_curve(path, conpoints)[0]

    ax2.plot(*path.T)
    ax2.plot(*conpoints_opt.T)
    plot_curve(conpoints_opt, ax2)
