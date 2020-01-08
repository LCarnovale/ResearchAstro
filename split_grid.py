import numpy as np

source = 'out.txt'

print("loading grid...")
table = np.genfromtxt(source, delimiter=',', skip_header=1)
full = {}

full['id'] = table[:,0]
full['mass'] = table[:,1]
full['Y'] = table[:,2]
full['feh'] = table[:,3]
full['alpha'] = table[:,4]
full['diff'] = table[:,5]
full['over'] = table[:,6]
full['teff'] = table[:,7]
full['lum'] = table[:,8]

class Track:
    def __init__(self, teff_ax, lum_ax, **init_vals):
        """ `init_vals` should be a dictionary of key:value pairs.
        """
        for k in init_vals:
            self.__setattr__(k, init_vals[k])
        
        self.T_ax = teff_ax
        self.L_ax = lum_ax
    
def get_track(mass, Y, feh, alpha, diff, over):
    # Try to filter the table:
    track = filter_rows(mass=mass, Y=Y, feh=feh, alpha=alpha, diff=diff, over=over)
    if track is None:
        print("No track found for the given values.")
        return None
    
    T_ax = track['teff']
    L_ax = track['lum']
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
    for k in source:
        if k in ['lum, teff']: continue
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
    for k in source:
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