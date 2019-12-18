import numpy as np
import matplotlib.pyplot as plt
import fit_Bezier as fb

import tools

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 10}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

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

def get_path(track, phase):
    """ Return the path between the given phase point and the next phase
    along the H-R path for the given track.
    """

    EEPs = track['EEPs']

    path_full = np.array([track['log_Teff'], track['log_L']]).T

    return path_full[EEPs[phase]:EEPs[phase + 1]]
    
# Each phase and the number of control points to fit with, 
# and the number of points to split the path into for fitting.
path_fits = [
    (1, 10, 500),
    (2, 10, 100),
    (3, 50, 500),
    (7, 50, 500),
    (8, 50, 100)
]

tracks = [
    get_track(1, 0) # sun
]

optimized_c_points = {}

fig = plt.figure("Fitting bezier curves to HR tracks")
ax = fig.subplots()
ax.invert_xaxis()

for track in tracks:
    for path_info in path_fits:
        phase, c_count, n_points = path_info
        path = get_path(track, phase)

        # Begin with control points evenly spaced out along the path.
        dists = np.linspace(0, 1, c_count)
        c_points = fb.interp_path(path, dists, normalised=True)
        # Get optimized control points.
        # plt.plot(*c_points.T, "ro")
        # plt.plot(*path.T)
        # plt.show()
        c_opt = fb.fit_curve(path, c_points, n_points)[0]

        optimized_c_points[phase] = c_opt
    
        ax.plot(*path.T, '.-', label="Phase %d path"%phase)
        p = fb.plot_curve(c_opt, ax, n_points)

plt.show()

# for track in tracks:
#     for i in range(7):
#         path = get_path()
        


