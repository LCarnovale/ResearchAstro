import numpy as np
import fit_Bezier as fb

from sys import argv

init_args = []
full = {}
def init(*args):
    global full
    global init_args
    init_args = args

    if 'big' in argv:

        source = [
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_0.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_1.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_2.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_3.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_4.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_5.txt",
            r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_6.txt",
        ][:]

        print("loading grid...")
        with open(source[0], 'r') as f:
            header_line = next(f)

        ## Assume all rows are comma separated ##

        headers = header_line.split(",")

        table = np.genfromtxt(source[0], delimiter=',', skip_header=1)
        if len(source) > 1:
            print("1 done")
            tables = [table]
            for s in source[1:]:
                new = np.genfromtxt(s, delimiter=',', skip_header=0)
                tables.append(new)
                print("loaded", s)

            table = np.concatenate(tables)

        full = {}

        for i, h in enumerate(headers):
            full[h] = table[:, i]
    else:
        import read_hist
        full = read_hist.get_full_cols(*args)

# full['id'] = table[:,0]
# full['mass'] = table[:,1]
# full['Y'] = table[:,2]
# full['feh'] = table[:,3]
# full['alpha'] = table[:,4]
# full['diff'] = table[:,5]
# full['over'] = table[:,6]
# full['teff'] = table[:,7]
# full['lum'] = table[:,8]

def keys(ob):
    try:
        return ob.keys()
    except:
        try:
            return ob.dtype.names
        except:
            raise Exception("Unable to get keys/names of obejct.")

class Track:
    def __init__(self, T_ax, L_ax, **init_vals):
        """ `init_vals` should be a dictionary of key:value pairs.

        T_ax and L_ax are expected to be log values.
        """
        for k in init_vals:
            self.__setattr__(k, init_vals[k])

        self.T_ax = T_ax
        self.L_ax = L_ax

        # Calculate some other useful things

        points = np.array([self.T_ax, self.L_ax]).T
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
        self.curv_inv_rad = inv_radius
        self.curv_norm = inv_radius / max(inv_radius)
        self.tangent = tangent
        self.path_len = dists

        try:
            self.avg_density = self.star_mass / (10**(3*self.log_R))
            self.log_avg_rho = np.log(track['avg_density'])
        except:
            pass


class TrackSet:
    def __init__(self, *tracks):
        self._tracks = tracks
        self._eps = [list() for _ in tracks]
        self._ep_funcs = []

    def __iter__(self):
        return self._tracks.__iter__()

    def __getitem__(self, item):
        return self._tracks[item]

    def __len__(self):
        """ Get the number of tracks in this set """
        return len(self._tracks)

    def add_ep_fun(self, func):
        """ Add a function to find ep (evolution points) points on tracks.
        When added, the function will be used on all current tracks,
        and all future tracks will be run by it aswell.

        func: func(track) -> index location in track.
        """
        for ep, t in zip(self._eps, self._tracks):
            ep.append(func(t))

        self._ep_funcs.append(func)

    def add_track(self, track):
        """ Add a track to the trackset. Any ep functions attached to this set
        will be run on the added track.
        """
        self._tracks.append(track)
        eps = []
        for f in self._ep_funcs:
            eps.append(f(track))
        self._eps.append(eps)

    def get_ep_point(self, track_index, ep_index):
        """ Get a (log_T, log_L) point for a track's ep point.

        track_index: the track number in this set.
        ep_index: the ep number for the track
        """
        t = self._tracks[track_index]
        ep = self._eps[track_index][ep_index]
        point = (t.T_ax[ep], t.L_ax[ep])
        return np.array(point)

    def get_ep_points(self, track_index):
        """ Get a list of (log_T, log_L) points for a track.
        """
        t = self._tracks[track_index]
        eps = self._eps[track_index]
        points = [(t.T_ax[ep], t.L_ax[ep]) for ep in eps]
        return np.array(points)

    @property
    def eps(self):
        "Evolutionary points for all tracks"
        return self._eps

    @property
    def num_eps(self):
        return len(self._ep_funcs)


def get_track_from_id(id, *extra_keys):
    """ Get a track from it's id
    Extra keys can be provided if the source is known to contain them.
    """
    mask = full['id'] == id

    init_mass =  full['M'][mask][0]
    init_Y =  full['Y'][mask][0]
    init_feh =  full['Z'][mask][0]
    init_alpha =  full['alpha'][mask][0]
    init_diff =  full['diffusion'][mask][0]
    init_over =  full['overshoot'][mask][0]

    extras = {e: full[e][mask] for e in extra_keys + init_args}

    T_ax = full['log_Teff'][mask]
    L_ax = full['log_L'][mask]

    return Track(T_ax, L_ax, mass=init_mass,
        Y=init_Y, feh=init_feh, alpha=init_alpha,
        diff=init_diff, over=init_over, **extras)

def get_track(mass, Y, feh, alpha, diff, over):
    # Try to filter the table:
    track = filter_rows(mass=mass, Y=Y, feh=feh, alpha=alpha, diff=diff, over=over)
    if track is None:
        print("No track found for the given values.")
        return None

    T_ax = track['log_Teff']
    L_ax = track['log_L']
    new_track = Track(T_ax, L_ax, mass=mass, Y=Y, feh=feh, alpha=alpha, diff=diff, over=over)
    return new_track

def get_unique_tracks(source=None):
    """ Get all possible unique tracks from the given table or from the
    default full table. Output table does not include teff or lum.

    Assumes that id's correspond to unique values for all columns other than teff and lum."""
    if source is None:
        source = full

    _, u_indexes = np.unique(source['id'], return_index=True)
    out = {}
    try:
        keys = source.keys()
    except:
        try:
            keys = source.dtype.names
        except:
            raise Exception("Unable to get names/keys of record object.")

    for k in keys:
        if k in ['log_Teff', 'log_L']: continue
        out[k] = source[k][u_indexes]

    return out



def get_vals(key, source=None):
    """ Get the unique values for a given key.
    `key` can be any of:

        id, mass, Y, feh, alpha, diff, over, teff, lum

    Default source is the full table. A smaller filtered source can
    be provided with `source`, which must be a dictionary with the keys
    shown above.
    """
    if source is None:
        source = full
    return np.unique(source[key])

def filter_rows(source=None, **kwargs):
    """ Return a dictionary with rows filtered down by the given filters.
    eg:
        >>> x = filter_rows(mass=1)
        >>> x['mass']
        [1, 1, 1, ..., 1, 1, 1]
        >>> x['lum']
        [0.2, 0.21, 0.22, ..., 0.54, 0.55, 0.56]
    etc.

    Default source is the full table, but a different table can be provided
    with `source`. It should have the same keys as the normal table.
    """
    if source is None:
        source = full

    mask = source['id'] > 0
    for k in kwargs:
        vals = source[k]
        mask = np.logical_and(mask, vals==kwargs[k])
        if not np.any(mask):
            print("The following filter returned no results:")
            print(kwargs)
            return None

    out = {}
    for k in keys(source):
        out[k] = source[k][mask]

    return out


def make_safe(s, rounding=None):
    """Make a string safe for use in a filename
    Replaces `.` with `_`, `+` with `p` and `-` with `m`

    A number can also be provided, and optionally rounded by also
    providing the number of desired decimal points under `rounding`."""

    if rounding is not None and type(s) is str:
        raise TypeError("s must be a number if rounding is specified.")

    if rounding:
        s = round(s, rounding)
    elif rounding == 0:
        s = round(s, 0)
        s = int(s)

    s = str(s)

    s = s.replace('.', '_')
    s = s.replace('+', 'p')
    s = s.replace('-', 'm')
    return s


# row_count = len(id_ax)

# start_flag = True
# for i in range(row_count):
#     if start_flag:
#         m = make_safe(mass_ax[i], 4)
#         Y = make_safe(Y_ax[i], 6)
#         Feh = make_safe(Feh_ax[i], 6)
#         alpha = make_safe(alpha_ax[i], 5)

#         # Start a new track
#         fname = f"m_{m}_Y_{Y}"


print("Done.")
