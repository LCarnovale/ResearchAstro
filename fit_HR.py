import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb

import tools


def get_track(mass, fe_h, a_fe=0, v_vcrit=0):
    """ Fetch the evolutionary track of a star.
    'Mass' is the initial mass, in solar masses.
    'fe_h' is the metalicity of the star, 0 for a sun like star.
    'a_fe' is the alpha to iron abundance, 0 for a sun like star.
    'v_vcrit' is the rotation of the star as a ratio of the critical rotation.
    MIST only includes rotation values of 0 and 0.4.

    Returns a dictionary with all 77 fields from the table if it can be found.
    """

    m_str = f"{int(mass*1e2):05d}"
    
    feh = (fe_h<0 and 'm') or (fe_h>=0 and 'p')
    feh = f'{feh}{abs(fe_h):.02f}'
    afe = (a_fe<0 and 'm') or (a_fe>=0 and 'p')
    afe = f'{afe}{abs(a_fe):.01f}'

    fname = f"MistTracks/MIST_v1.2_feh_{feh}_afe_{afe}_vvcrit{v_vcrit:.01f}_EEPS/{m_str}M.track.eep"
    print(f"Fetching: {fname}")
    try:
        data = np.genfromtxt(fname)    # The table data.
        data = data.T
        # Get headers etc.
        with open(fname, 'r') as f:
            lines = f.readlines()
            headers = lines[11][1:-1]
            remove_spaces = lambda x: [i for i in x.split(' ') if i]
            headers = remove_spaces(headers)

            EEPs = lines[8][2:-1]
            EEPs = [int(x)-1 for x in EEPs[5:].split(' ') if (x and int(x) <= len(data[0]))]

            inits = remove_spaces(lines[3][1:-1])
            inits += remove_spaces(lines[6][1:-1])
            init_vals = remove_spaces(lines[4][1:-1])
            init_vals += remove_spaces(lines[7][1:-1])
            init_vals = [float(x) for x in init_vals[:-2]] + init_vals[-2:]

    except FileNotFoundError:
        print("No track found for:")
        print("mass:", mass)
        print("fe_h:", fe_h)
        print("a_fe:", a_fe)
        print("v_vcrit:", v_vcrit)
        return None
    else:
        pass
        
    out = dict()
    
    for init, val in zip(inits, init_vals):
        out[init] = val
    
    out["EEPs"] = EEPs

    for i, h in enumerate(headers):
        out[h] = data[i]
    
    
    return out
    
def get_phase_points(*args):
    """ 
    Usage:
        get_phase_points(mass, fe_h)
        get_phase_points(track)

    Return an array of Temp-Lum pairs representing the
    positions of evolutionary points along the requested track.
    """
    if len(args) >= 2:
        track = get_track(*args[:2])
    else:
        track = args[0]

    E_ax = track['EEPs'][:]
    if len(E_ax) <= 7:
        print("this one bad: M and Fe/H =", track['initial_mass'], track['[Fe/H]'])

    t_ax = track['log_Teff']
    L_ax = track['log_L']

    # Try to find 'local inflexion' points
    rise = L_ax[1:] - L_ax[:-1]
    run  = t_ax[1:] - t_ax[:-1]
    gradient = rise / run

    # Select phase start points of a section (ie the EEP number)
    # to analyse. 
    slices_to_check = [slice(3,5), slice(6,4,-1), slice(9, 7, -1)]
    for slic in slices_to_check:
        rise = np.diff(L_ax[E_ax[slic]])[0]
        run  = np.diff(t_ax[E_ax[slic]])[0]
        phase_grad = rise / run

        idx, idx_next = E_ax[slic]

        # Cut out the section to be analysed
        section = gradient[idx:idx_next:(-1 if idx>idx_next else 1)]
        diff = section - phase_grad
        # Negate the diff if the initial slope is positive
        diff[2] < 0 or diff.__imul__(-1)
        try:
            min_idx = tools.getFromTrigger(diff, 0)[0]
        except Exception as e:
            plt.plot(diff)
            plt.show()
            raise e
        if (idx > idx_next):
            # Swap if we are looking at a reversed slice
            idx, idx_next = idx_next, idx

        # Add the new inflexion point
        E_ax = np.append(E_ax, [min_idx+idx])
        # E_ax = [*E_ax[:phase+1], min_idx + idx, *E_ax[phase+1:]]

    
    E_T = t_ax[E_ax]
    E_L = L_ax[E_ax]

    return np.array([E_T, E_L]).T

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

generated = {}
def generate_matrix(N):
    """ Generate a coeffecient matrix of size `N` for the calculation
    of Bezier curves. This returns `M` in `B(t) = T(t) * M * P`."""
    global generated
    if N in generated:
        return generated[N]
    else:
        rows = np.array([[row for col in range(N)] for row in range(N)])
        cols = np.array([[col for col in range(N)] for row in range(N)])
        M = A(rows, cols, N)
        generated[N] = M
        return M
    

    

def T(t, n):
    """ Return an array of `[1, t^1, t^2, ..., t^(n-1)]`
    for every value of `t`. `t` can be scalar or a 1-D array.
    """
    a = np.array([t**i for i in range(n)]).T
    size = np.size(t)
    if size > 1:
        return a.reshape(size, -1, 1)
    else:
        return a.reshape(-1, 1)

def curve_eval(x, control_points):
    """ `control_points` should be a N x 2 array.
    x should be an array like of floats between 0 and 1. """
    N = len(control_points)
    M = generate_matrix(N)
    Tm = T(x, N)

    points = (Tm*M).dot(control_points)
    points = points.sum(axis=1)
    return points

def plot_curve(control_points):
    t_ax = np.linspace(0, 1, 100)
    points = curve_eval(t_ax, control_points)
    return plt.plot(*points.T)

# Fit somehow???????????????

track = get_track(1, 0, 0, 0.4)

EEPs = get_phase_points(track)
