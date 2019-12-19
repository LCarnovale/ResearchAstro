import numpy as np
import matplotlib.pyplot as plt
import fit_Bezier as fb

import tools

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 15}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

# class TrackList:
#     def 
class TrackSet:
    _valid_vars = ['mass', 'feh', 'v']
    # _var_keys   = ['initial_mass', '[Fe/H]', '[a/Fe]', 'v/vcrit']
    def __init__(self, independent_var, vals, defaults, indep_str=None):
        """Fetch a set of tracks with a varying value for `independent_var`,
        which must be either 'mass', 'feh', or 'v'.
        `vals` should be the values that the independent variable should take,
        and defaults should contain a value for all of the above parameters 
        (in that order), and will determine the values of the fixed parameters.
        The value in the defaults corresponding to the independent variable will
        be ignored.

        Instantiating objects of this class does not fetch the tracks immediately.
        To do so, call init() on the newly created object. This is simply so that
        many track sets can be established quickly, and loaded when needed.

        Each of the defaults other than the independent variable can be accessed
        by: 
            >>> ts = TrackSet('feh', [1, 2, 3], (1, None, 2))
            >>> ts.mass
            1
        Accessing the independent variable will return the list of values
        that variable takes across all tracks.
            >>> ts.feh
            [1, 2, 3]

        if `indep_str` is specified, then calling `ts.strfmt(track)` or 
        `ts.strfmt((int) i)` will return `indep_str.format(<indep var value>)` for
        the given track or the i'th track. This can be used for formatting values with units.
        If `indep_str` is not specified, ts.strfmt() will just return the corresponding
        value as is. 
        """
        if independent_var not in self._valid_vars:
            raise ValueError('Invalid independent variable: %s' % independent_var)
        
        if indep_str == None:
            self.indep_str = "{}"
        else:
            self.indep_str = indep_str
        
        # Get the tracks
        self._tracks = []
        # for v in vals:
        #     args = [i for i in defaults] # get a copy of defaults
        #     args[idx] = v
        #     self._tracks.append(get_track(*args))
        
        self.indep_var = independent_var
        self.vals = vals
        self.defaults = defaults
        
        for var, val in zip(self._valid_vars, defaults):
            if var == independent_var:
                self.__setattr__(var, vals)
            else:
                self.__setattr__(var, val)

    def __iter__(self):
        return iter(self._tracks)

    def __getitem__(self, var):
        return self._tracks.__getitem__(var)

    def init(self):
        """ Fetch all the tracks. You need to call this to be able
        to do anything with this object."""
        idx = self._valid_vars.index(self.indep_var)
        for v in self.vals:
            args = [i for i in self.defaults] # get a copy of defaults
            args[idx] = v
            self._tracks.append(get_track(*args))

    def strfmt(self, arg):
        """
        `arg`: A track in the set or an integer as the index of some track.
        Return a string representation of the independent variable in that track.
        """
        if type(arg) is not int:
            arg = self._tracks.index(arg)

        return self.indep_str.format(self.vals[arg])
        
    




        


def get_track(mass, fe_h, v_vcrit=0):
    """ Fetch the evolutionary track of a star.
    'Mass' is the initial mass, in solar masses.
    'fe_h' is the metalicity of the star, 0 for a sun like star.
    'v_vcrit' is the rotation of the star as a ratio of the critical rotation.
    MIST only includes rotation values of 0 and 0.4.

    Returns a dictionary with all 77 fields from the table if it can be found.
    """

    m_str = f"{int(round(mass*1e2)):05d}"
    
    feh = (fe_h<0 and 'm') or (fe_h>=0 and 'p')
    feh = f'{feh}{abs(fe_h):.02f}'

    fname = f"MistTracks/MIST_v1.2_feh_{feh}_afe_p0.0_vvcrit{v_vcrit:.01f}_EEPS/{m_str}M.track.eep"
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

def interp_track(track, key, value, domain=None, left=None, right=None):
    """ Return a dict matching the keys of the given track with all values
    interpolated to correspond with the given `value` for the given `key`.

    Provide a slice to interpolate over with `domain` to avoid bugs due to
    some columns not being strictly increasing. Default is `slice(None)`,
    ie the whole column will be used for interpolation.
    """

    if domain == None:
        domain = slice(None)

    x_ax = track[key][domain]

    out = dict()
    for k in track:
        if k == key:
            out[k] = value
        else:
            try:
                out[k] = np.interp(value, x_ax, track[k][domain], left=left, right=right)
            except:
                # If it isn't an array, don't interpolate
                out[k] = track[k]
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
    slices_to_check = [slice(3,5)]
    for slic in slices_to_check:
        rise = np.diff(L_ax[E_ax[slic]])[0]
        run  = np.diff(t_ax[E_ax[slic]])[0]
        phase_grad = rise / run

        idx, idx_next = E_ax[slic]

        # Cut out the section to be analysed
        section = gradient[idx:idx_next:(-1 if idx>idx_next else 1)]
        diff = section - phase_grad
        # Negate the diff if the initial slope is positive
        diff[1] < 0 or diff.__imul__(-1)
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

    track['EEPs'] = np.sort(E_ax)
    E_T = t_ax[track['EEPs']]
    E_L = L_ax[track['EEPs']]

    return np.array([E_T, E_L]).T

def get_path(track, phase):
    """ Return the path between the given phase point and the next phase
    along the H-R path for the given track.
    """

    EEPs = track['EEPs']

    path_full = np.array([track['log_Teff'], track['log_L']]).T

    return path_full[EEPs[phase]:EEPs[phase + 1]]
    


def get_track_fits(track):
    """ Return a function that will give the approximated
    position of a point some distance along a given phase.

    Use the returned function as:
        f = get_track_fits(track)
        p = f(1, 0.25)
    To get the location of the point between the 1st and 2nd EEP
    at 25% the total distance along the matching bezier curve.
    """
    
    # Fit the function
    global optimized_c_points
    all_c_points = dict()
    
    for path_info in path_fits:
        phase, c_count, n_points = path_info
        print("Fitting phase %d" % phase)
        path = get_path(track, phase)

        # Begin with control points evenly spaced out along the path.
        dists = np.linspace(0, 1, c_count)
        # c_points = fb.interp_path(path[np.array([0, -1])], dists, normalised=True)
        c_points = fb.interp_path(path, dists, normalised=True)
        # Get optimized control points.
        # plt.plot(*path.T)
        # plt.show()
        c_opt = fb.fit_curve(path, c_points, n_points)[0]
        # print(c_opt)
        # plt.plot(*c_opt.T, "o", label="Phase %d control points"%phase)
        all_c_points[phase] = c_opt.copy()
        # def temp_f(t, c):
        #     temp_c = c_opt.copy()
        #     return fb.curve_eval(t, temp_c)
        
        # all_c_points[str(phase)] = temp_f
        
    # optimized_c_points[track] = all_c_points

    def func(n, t):
        if n in all_c_points:
            cpoints = all_c_points[n]
            d_to_t = fb.get_inverse_dist_func(cpoints)
            return fb.curve_eval(d_to_t(t), cpoints)
        else:
            raise ValueError("The given phase has not been fit to (%d)"%n)
        
    return func



tracks_m_var = TrackSet('mass',
    [.8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
    (None, 0, 0), indep_str=r"{} $M_\odot$"
)

tracks_feh_var = TrackSet('feh', 
    [-2.0, -1.75, -1.5, -1.25, -1.0, 
     -0.75, -0.5, -0.25, 0.0, 0.25, 0.5], 
     (1, None, 0), indep_str="[Fe/H]: {}"
)

tracks_v_var = TrackSet('v',
    [0, 0.4], (1, 0, None))


# for track in tracks:
#     get_phase_points(track)

# Each phase and the number of control points to fit with, 
# and the number of points to split the path into for fitting.
path_fits = [
    (1, 22, 100),
    (2, 10, 100),
    (3, 8, 400),
    # (4, 20, 100),
    (5, 5, 100),
    (8, 20, 100)
]

optimized_c_points = {}

fig = plt.figure("Fitting bezier curves to HR tracks")
ax = fig.subplots()
ax.invert_xaxis()

# track_fs = [get_track_fits(track) for track in tracks]


contour_key = 'log_center_Rho'

n_arrows_all = {
    1: 5,
    2: 10,
    3: 20,
    5: 50,
    8: 0
}

tracks = tracks_v_var
tracks.init()

for phase, _, _ in path_fits:
    # plot the lines
    points = []
    cont_points = [] # contour points, not control points
    n_arrows = n_arrows_all[phase]
    t_ax = np.linspace(0, 1, n_arrows)
    for i, track in enumerate(tracks):
        path = get_path(track, phase)
        domain = slice(*track['EEPs'][phase:phase + 2])
        if i == 0:
            contour_range = track[contour_key][domain]
            # start_row = interp_track(track, 'log_Teff', path[0][0], domain)
            # end_row   = interp_track(track, 'log_Teff', path[-1][0], domain)
            contour_vals = np.linspace(
                min(contour_range)*.9, 1.1*max(contour_range), 
                n_arrows+1, endpoint=False
            )[1:]

        contour_points = interp_track(track, contour_key, contour_vals, domain, left=0, right=0)
        contour_points = np.array([contour_points['log_Teff'], contour_points['log_L']]).T
        cont_points.append(contour_points)

        # if phase == 0:
        plt.plot(*path.T, color="C%d"%(phase%10), label=(None if phase>1 else tracks.strfmt(i)))
        # else:
        #     plt.plot(*path.T, color="C%d"%(phase%10), label=None)

        # c_points = optimized_c_points[track][phase]

        # points.append(track_fs[i](phase, t_ax)) # Use Bezier curve fits
        points.append(fb.interp_path(path, t_ax, normalised=True)) # Interpolate over the path
    
    points = np.array(points)
    cont_points = np.array(cont_points)

    for p in (cont_points, ):    
        diff = p[1:] - p[:-1]
        diff[p[1:] == 0.] = 0.
        diff[p[:-1] == 0.] = 0.
        plt.quiver(
            p[:-1,:,0], p[:-1,:,1], diff[:,:,0], diff[:,:,1], 
            scale=1, angles='xy', scale_units='xy', color='gray',#"C%d"%(phase%10),
            width=0.001, label=None#"Phase %d change"%phase  
        )

        # p = fb.plot_curve(c_opt, ax, len(path))[0]
        # p.set_label("Phase %d fit"%phase)
        # p.set_linestyle('--')
plt.title(f"Evolutionary Tracks for varying {tracks.indep_var}, lines of constant {contour_key}")
plt.xlabel(r"$\log(T_{eff})$")
plt.ylabel(r"$\log(L)$")
plt.legend()
plt.show()


        


