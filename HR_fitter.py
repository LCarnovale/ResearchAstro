import numpy as np
import matplotlib.pyplot as plt
import fit_Bezier as fb

import tools

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 25}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

class TrackSet:
    _valid_vars = ['mass', 'feh', 'v', 'alpha', 'Y', 'overshoot', 'diffusion']
    # _var_keys   = ['initial_mass', '[Fe/H]', '[a/Fe]', 'v/vcrit']
    def __init__(self, independent_var, vals, defaults, indep_str=None):
        """Fetch a set of tracks with a varying value for `independent_var`,
        which must be either 'mass', 'feh', 'v', 'alpha', 'Y', 'overshoot', or 'diffusion'.
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

    def __len__(self):
        return len(self._tracks)

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
        
    



def contour_1d(data, contour_step, start=None):
    """ Returns a list of interpolated indexes, and 
    step distances from the start contour, for locations of 
    values separated by `contour_step` in `data`. 

    A base or start contour value can be set with `start`
    (by default the first value in  `data`)

    Eg:
        >>> data = [2, 3, 4, 5, 6, 7, 8, 8.5, 9, 10]
        >>> contour_1d(data, 2, start=6)
        [(0, -2), (2, -1), (4, 0), (6, 1), (9, 2)]
    If the exact contour is not in data, the index is interpolated:
        >>> data = [2, 3, 4, 5, 6, 7, 8, 8.5, 9, 11]
        >>> contour_1d(data, 2, start=6)
        [(0, -2), (2, -1), (4, 0), (6, 1), (8.5, 2)]
    """
    
    if start == None:
        start = data[0]
    
    vals = data - start

    steps = vals // contour_step

    mask = steps[1:] != steps[:-1] 
    # mask indicates the intervals between data points
    # where contours lie.
    cont_intervals = np.flatnonzero(mask)
    # for each i in cont_intervals, vals[i]->vals[i+1]
    # is an interval with contours on it.
    out = []
    if not vals[0] % contour_step:
        out.append([0, steps[0]])

    for interval in cont_intervals:
        i0, i1 = interval, interval+1
        # Get the data values at each end
        f0, f1 = vals[i0:i0+2]
        # Get the values of contours within the relevant steps
        order = int(np.sign(f1 - f0))
        s0, s1 = steps[i0:i0+2]
        s_ax = np.linspace(*[s1, s0][::order], abs(s1-s0), endpoint=False)
        c_ax = s_ax * contour_step
        # interpolate to get indexes
        idxs = np.interp(c_ax, [f0, f1][::order], [i0, i1][::order],
            left=None, right=None)

        for i, s in zip(idxs, s_ax):
            if i is None: continue
            out.append([i, s])
            
    return np.array(out)
    
# data = np.arange(10)-4
# print(contour_1d(data, 3, start=2))

# xax = np.linspace(-5, 5, 100)
# yax = xax**2

# conts = contour_1d(yax, 5, start=0)
# conts = np.array(conts)
# plt.plot(xax, yax, '.-')
# c_x_ax = np.interp(conts[:, 0], np.arange(len(xax)), xax)
# c_y_ax = np.interp(conts[:, 0], np.arange(len(xax)), yax)
# c_step_ax = conts[:, 1] * 5
# plt.plot(c_x_ax, c_y_ax, "rx")
# plt.plot(c_x_ax, c_step_ax, "k+")
# plt.legend(['data', 'interped y', 'stepped y'])
# plt.show()

# raise Exception()

def get_track(mass, fe_h, v_vcrit=0, alpha=None, Y=None, diff=None, over=None):
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

def interp_track(track, key, value, domain=None, left=None, right=None, index_axis=False):
    """ Return a dict matching the keys of the given track with all values
    interpolated to correspond with the given `value` for the given `key`.

    Provide a slice to interpolate over with `domain` to avoid bugs due to
    some columns not being strictly increasing. Default is `slice(None)`,
    ie the whole column will be used for interpolation.

    `index_axis` default False, if True then an index of the value in the track
    is found by interpolation, and then the other fields are interpolated with
    the indexes (0, 1, 2, ...) as the x axis, and the found index as the value. 
    When False, the values of the key field are used as the x axis.
    """

    if domain == None:
        domain = slice(None)

    if index_axis:
        y_ax = np.arange(len(track[key]))
        idx = np.interp(value, track[key][domain], y_ax[domain])
        x_ax = y_ax
        value = idx
        domain = slice(None)
    else:
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
    
def analyse_track(*args):
    """ 
    Usage:
        analyse_track(mass, fe_h)
        analyse_track(track)

    Add some useful fields to the track like radius of curvature,
    path length, tangents, gradients etc.

    """
    if len(args) >= 2:
        track = get_track(*args[:2])
    else:
        track = args[0]


    # Get distance array:
    T_ax = track['log_Teff']
    L_ax = track['log_L']
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

    # Get points of minimum curvature
    track['curv_inv_rad'] = inv_radius
    track['curv_norm'] = inv_radius / max(inv_radius)
    track['tangent'] = tangent
    track['path_len'] = dists

    track['avg_density'] = track['star_mass'] / (10**(3*track['log_R']))
    track['log_avg_rho'] = np.log(track['avg_density'])

    track['idx'] = np.arange(len(dists))

    E_ax = track['EEPs'][:]

    # Put a point at the kink near the 1st segment
    start_idx = E_ax[1]
    curv_ax_rev = inv_radius[E_ax[2]:E_ax[1]:-1]
    end_idx = tools.getFromTrigger(curv_ax_rev, 100)[0]
    end_idx = len(curv_ax_rev) - 1 - end_idx # index was from the reversed segment
    end_idx += start_idx
    E_ax = np.array([*E_ax[:2], end_idx, *E_ax[2:]])
    
    if len(E_ax) <= 7:
        print("this one bad: M and Fe/H =", track['initial_mass'], track['[Fe/H]'])

    # Try to find 'local inflexion' points
    rise = L_ax[1:] - L_ax[:-1]
    run  = T_ax[1:] - T_ax[:-1]
    gradient = rise / run



    # Select phase start points of a section (ie the EEP number)
    # to analyse. 
    slices_to_check = []#slice(3,5)]
    for slic in slices_to_check:
        start, end = E_ax[slic]
        start += 25 # Fudge fix
        domain = slice(start, end)
        min_curv = np.argmax(inv_radius[start:end])
        print(track['[Fe/H]'], "sharpest point at", points[min_curv + start])
        print("Curvature inverse:", inv_radius[min_curv + start])
        print(f"Index: {min_curv} + {start} = {min_curv + start}, end of slice: {end}")
        E_ax = np.append(E_ax, [min_curv+start])
        # rise = np.diff(L_ax[E_ax[slic]])[0]
        # run  = np.diff(T_ax[E_ax[slic]])[0]
        # phase_grad = rise / run

        # idx, idx_next = E_ax[slic]

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

    # Calculate normalised path length


    track['EEPs'] = np.sort(E_ax)

    divisions = len(E_ax)
    # for p in track['EEPs']:
        # Get the path length array between here and the next point
    
    E_T = T_ax[track['EEPs']]
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
    # [.8, .9, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
    [.8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
    (None, 0, 0), indep_str=r"{} $M_\odot$"
)

tracks_feh_var = TrackSet('feh', 
    [-2.0, -1.75, -1.5, -1.25, -1.0, 
     -0.75, -0.5, -0.25, 0.0, 0.25, 0.5], 
     (1, None, 0), indep_str="[Fe/H]: {}"
)

tracks_v_var = TrackSet('v',
    [0, 0.4], (0.8, 0, None))


# Each phase and the number of control points to fit with, 
# and the number of points to split the path into for fitting.
path_fits = [
    (2, 22, 100),
    (3, 10, 100),
    (4, 8, 400),
    # (4, 20, 100),
]

optimized_c_points = {}

fig = plt.figure("Fitting bezier curves to HR tracks")
ax, ax2 = fig.subplots(ncols=2)
ax2.invert_xaxis()
ax.invert_xaxis()

contour_key = 'log_avg_rho'
normalise_contour_data = False # Normalise data before finding contours

n_arrows_all = {
    1: 20,
    2: 5,
    3: 10,
    4: 50,
    5: 10,
    6: 10,
}

import sys
argv = sys.argv
if 'm' in argv:
    tracks = tracks_m_var
elif 'feh' in argv:
    tracks = tracks_feh_var
else:
    print('using mass variation tracks, give "m" or "feh" for a specific track')
    tracks = tracks_m_var
tracks.init()

quiver_kwargs = {
    'angles':'xy', 
    'scale_units':'xy', 
    'scale':1
}

for track in tracks:
    analyse_track(track)

for phase, _, _ in path_fits:
    # plot the lines
    points = []
    cont_points = [] # contour points, not control points
    n_arrows = n_arrows_all[phase]
    t_ax = np.linspace(0, 1, n_arrows)
    print("phase:", phase)
    phase_colour = "C%d"%(phase%10)
    l_pads = []
    for i, track in enumerate(tracks):
        print("track:", tracks.strfmt(i))
        # path = np.array([track['log_Teff'], track['log_L']]).T
        NUM_SEG = 1 # Number of segments to plot over 
        domain = slice(*track['EEPs'][phase:phase + NUM_SEG+1:NUM_SEG])
        # domain = slice(track))
        contour_data = track[contour_key][domain] # Values to use to find contours
        if normalise_contour_data:
            contour_data /= max(contour_data)
        if i == 0:
            # start_row = interp_track(track, 'log_Teff', path[0][0], domain)
            # end_row   = interp_track(track, 'log_Teff', path[-1][0], domain)
            # contour_vals = np.linspace(
                # min(contour_data), max(contour_data), 
                # n_arrows+1, endpoint=False
            # )
            contour_data_range = max(contour_data) - min(contour_data)
            contour_step = contour_data_range / (n_arrows + 1)
            contour_step = contour_step
            middle = min(contour_data) + contour_data_range / 2
            zero_contour = middle

        contours = contour_1d(contour_data, contour_step, start=zero_contour)
        # if i > 0:
            
        # last_contour = contours
        
        if len(contours) == 0:
            contours = np.array([[0, 0]])
        idxs = contours[:, 0] + domain.start
        steps = contours[:, 1]
        print(steps)

        contour_points = interp_track(track, 'idx', idxs, left=0, right=0)
        contour_points = np.array([contour_points['log_Teff'], contour_points['log_L']]).T

        # if 1: #0 in steps:
            # Remove repeated steps:
        mask = np.array([True, *(steps[1:] != steps[:-1])])
        steps = steps[mask]
        contour_points = contour_points[mask]
        l_pad = int(120 - steps[0])
        l_pads.append(l_pad)
        if l_pad < 0:
            print("Bad pad!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        contour_points = np.pad(contour_points, [(l_pad, 0), (0, 0)], 'constant', constant_values=0)
        print("padding by", l_pad)
        

        contour_points = np.array(contour_points)

        cont_points.append(contour_points)
        # path = np.concatenate([get_path(track, p) for p in range(1, 8)])
        path = get_path(track, phase)

        # if phase == 0:
        ax.plot(*path.T, ':', color=phase_colour, label=(None if phase>2 else tracks.strfmt(i)))
        # last_phase = phase == path_fits[-1][0]
        if phase == path_fits[-1][0]:
            if i == 0 or i == (len(tracks) - 1):
                s = tracks.strfmt(i)
                data_point = path[-1]
                text_point = data_point + np.array([0., 0.1])
                ax.annotate(
                    s, data_point, xytext=text_point,
                    size='small', ha='center')
            

    max_len = max([len(l) for l in cont_points])
    for i in range(len(cont_points)):
        l = len(cont_points[i])
        cont_points[i] = np.pad(cont_points[i], [(0, max_len-l), (0, 0)], 'constant', constant_values=0)
    
    
    
    points = np.array(points)
    cont_points = np.array(cont_points)

    for p in (cont_points, ):
        diff = p[1:] - p[:-1]
        diff[p[1:] == 0.] = 0.
        diff[p[:-1] == 0.] = 0.
        diff_x = diff[:,:,0].flatten()
        diff_y = diff[:,:,1].flatten()

        # ax.quiver(
        #     p[:-1,:,0].flatten(), p[:-1,:,1].flatten(), diff_x, diff_y, 
        #     **quiver_kwargs, color='gray',#"C%d"%(phase%10),
        #     width=0.001, label=None#"Phase %d change"%phase  
        # )
        # Get the average diff
        for i in range(len(tracks) - 1):
            px = p[i,:,0]
            py = p[i,:,1]
            dx = diff[i,:,0]
            dy = diff[i,:,1]
            if tracks.indep_var == 'mass':
                mask = np.logical_and(dx > 0, dy > 0)
            else:
                mask = px != 0. 
            print(tracks.indep_var)
            px = px[mask]
            py = py[mask]
            dx = dx[mask]
            dy = dy[mask]
            
            samples = 5
            points_start = 1
            
            px = px[points_start::samples]
            py = py[points_start::samples]
            
            sum_x = dx[::samples]
            sum_y = dy[::samples]
            for i in range(1, samples):
                new_x = dx[i::samples]
                new_y = dy[i::samples]
                try:
                    sum_x += new_x
                    sum_y += new_y
                except:
                    sum_x[:-1] += new_x
                    sum_y[:-1] += new_y
            
            if len(sum_x) != len(px):
                sum_x = sum_x[:len(px)]
                sum_y = sum_y[:len(px)]
                
            sum_x /= samples
            sum_y /= samples
            
            ax.quiver(px, py, sum_x, sum_y, **quiver_kwargs, color=phase_colour, width=0.004)
            ax2.quiver(0, 0, sum_x, sum_y, **quiver_kwargs, color=phase_colour, width=0.002)
            
        sum_x = np.sum(diff[:, :, 0], axis=1)
        nnz_x = np.count_nonzero(diff[:, :, 0], axis=1)
        avg_x = sum_x / nnz_x
        sum_y = np.sum(diff[:, :, 1], axis=1)
        nnz_y = np.count_nonzero(diff[:, :, 1], axis=1)
        avg_y = sum_y / nnz_y
        
        avg_diff = np.array([avg_x, avg_y])
        # ax2.quiver(0, 0, avg_x, avg_y, **quiver_kwargs, color=phase_colour, width=0.002)
        # ax2.plot(avg_x, avg_y, 'o', color=phase_colour)
    # for i, track in enumerate(tracks):
        # if i == len(avg_x): continue
        # path = get_path(track, phase)
        # points = fb.interp_path(path, [.25, .5, .75], True)
        # points_x, points_y = points.T
        # ax.quiver(points_x, points_y, 
            # np.tile(avg_x[i], len(points_x)), 
            # np.tile(avg_y[i], len(points_x)), 
            # **quiver_kwargs, width=0.002)
      
        
ax.set_title(f"Evolutionary Tracks for varying {tracks.indep_var}, lines of constant {contour_key}")
ax.set_xlabel(r"$\log(T_{eff})$")
ax.set_ylabel(r"$\log(L)$")
ax2.set_title("Spread of track shifts")
ax2.set_xlabel(r"$\log(T_{eff})$")
ax2.set_ylabel(r"$\log(L)$")
ax2.plot(0, 0, 'ko')
ax.set_xlim(4, 3.4)
# ax.legend()
plt.show()


        


