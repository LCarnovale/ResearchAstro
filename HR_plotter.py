import numpy as np
import matplotlib.pyplot as plt
import sys

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 10}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

# Surface composition keys:
X_surf = 'surface_h1'
Y_surf = 'surface_he4'
# Core composition keys:
X_core = 'center_h1'
Y_core = 'center_he4'



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
            if (EEPs[0:4] != "EEPs"): print("EEPs couldn't be found")
            else:
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
    out["EEPs"] = EEPs
    for i, h in enumerate(headers):
        out[h] = data[i]
    
    for init, val in zip(inits, init_vals):
        out[init] = val
    
    return out

feh_vals = [.25, 0, -.25, -.5, -.75, -1.0, -1.25, -1.5, -1.75, -2]

tracks_feh_def = [
    get_track(1, f) for f in feh_vals
]

M_vals = [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

tracks_M_def = [
    get_track(m, 0) for m in M_vals
]


def get_phase_points(*args):
    """ Usage:
        get_phase_points(mass, fe_h)
        get_phase_points(track)
    Return an array of Temp-Lum pairs representing the
    positions of evolutionary points along the requested track.
    """
    if len(args) > 2:
        track = get_track(*args[:2])
    else:
        track = args[0]

    E_ax = track['EEPs'][:7]
    if len(E_ax) != 7:
        print("this one bad: M and Fe/H =", track['initial_mass'], track['[Fe/H]'])
    t_ax = track['log_Teff']
    L_ax = track['log_L']

    E_T = t_ax[E_ax]
    E_L = L_ax[E_ax]

    return np.array([E_T, E_L]).T

all_tracks = [
    [
        get_track(mass, fe_h) for mass in M_vals
    ] for fe_h in feh_vals
]

# To use this: all_phase_points[feh val][mass] -> list of points along evolution track
all_phase_points = np.array([
    [
        get_phase_points(track) for track in mass_ax
    ] for mass_ax in all_tracks
])

def plot_with_phases(tracks, ax, change_key, change_str, color_start=0, arrow_shape='full'):
    """Plot the tracks on a HR diagram,
    draw arrows connecting evolutionary points on each track,
    show a legend that will look like:
        change_str.format(track[change_key])

    for each track.
    """
    sun_style = '--'

    phase_points = []   # Will contain points along evolutionary tracks 
                        # at corresponding EEP points for each track

    ci = color_start
    for track in tracks:
        E_ax = track['EEPs']
        t_ax = track['log_Teff']
        L_ax = track['log_L']
        xi = E_ax[1] # Start point, will usually correspond with ZAMS phase
        if abs(track['initial_mass'] - 1) < 1e-3 and track['[Fe/H]'] == 0:
            _style = sun_style
        else:
            _style = ','
        p = ax.plot(t_ax[xi:], L_ax[xi:], _style, color="C%d"%(ci%10))
        c = p[0].get_color()
        EP_t_ax = t_ax[E_ax[1:]]
        EP_L_ax = L_ax[E_ax[1:]]
        phase_points.append(np.array([EP_t_ax, EP_L_ax]).T)
        ax.plot(EP_t_ax, EP_L_ax, '.', color=c, label=change_str.format(track[change_key]))
        ci += 1

    for phase in (1, 3, 6):
        print(phase)
        points = all_phase_points[:, :, phase].copy()
        points_fe_var = points[1:] - points[:-1]
        points_m_var  = points[:, 1:] - points[:, :-1]

        points_fe = points[:-1]
        fe_x = points_fe[:, :, 0].flatten()
        fe_y = points_fe[:, :, 1].flatten()
        fe_var_x = points_fe_var[:, :, 0].flatten()
        fe_var_y = points_fe_var[:, :, 1].flatten()
        
        points_m  = points[:, :-1]
        m_x = points_m[:, :, 0].flatten()
        m_y = points_m[:, :, 1].flatten()
        m_var_x = points_m_var[:, :, 0].flatten()
        m_var_y = points_m_var[:, :, 1].flatten()

        ax.quiver(fe_x, fe_y, fe_var_x, fe_var_y, 
            angles='xy', scale_units='xy', scale=1, color='red',
            width=0.002, label=f"-0.25 [Fe/H] starting from {feh_vals[0]}")
        ax.quiver(m_x, m_y, m_var_x, m_var_y, 
            angles='xy', scale_units='xy', scale=1, color='green',
            width=0.002, label=rf"+0.1 $M_\odot$ starting from {M_vals[0]}")




    # for phase in range(len(E_ax) - 1):
    #     for i in range(len(phase_points) - 1):
    #         try:
    #             this = phase_points[i][phase]
    #             next = phase_points[i + 1][phase]
    #         except:
    #             continue
    #         else:
    #             ax.arrow(*this, *(next - this), 
    #                 color='C%d'%((color_start + i)%10), 
    #                 length_includes_head=True,
    #                 shape=arrow_shape)

    ax.legend()

fig = plt.figure("HR Plot")
# ax1, ax2 = fig.subplots(nrows=2)
ax1 = fig.subplots()
ax1.set_xlabel(r'$\log(T_{eff})$')
ax1.set_ylabel(r'$\log(L)$')
ax1.invert_xaxis()
# ax2.set_xlabel(r'$\log(T_{eff})$')
# ax2.set_ylabel(r'$\log(L)$')
# ax2.invert_xaxis()


plot_with_phases(tracks_feh_def, ax1, '[Fe/H]', '[Fe/H]: {}', arrow_shape='left')
plot_with_phases(tracks_M_def, ax1, 'initial_mass', r'Start mass: ${} M_\odot$', arrow_shape='right')

plt.show()