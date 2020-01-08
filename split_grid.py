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

def filter_rows(**kwargs):
    """ Return a dictionary with rows filtered down by the given filters.
    eg:
        >>> x = filter_rows(mass=1)
        >>> x['mass']
        [1, 1, 1, ..., 1, 1, 1]
        >>> x['lum']
        [0.2, 0.21, 0.22, ..., 0.54, 0.55, 0.56]
    etc.
    """
    mask = full['id'] > 0
    for k in kwargs:
        vals = full[k]
        mask = np.logical_and(mask, vals==kwargs[k])

    out = {}
    for k in full:
        out[k] = full[k][mask]

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


row_count = len(id_ax)

start_flag = True
for i in range(row_count):
    if start_flag:
        m = make_safe(mass_ax[i], 4)
        Y = make_safe(Y_ax[i], 6)
        Feh = make_safe(Feh_ax[i], 6)
        alpha = make_safe(alpha_ax[i], 5)

        # Start a new track
        fname = f"m_{m}_Y_{Y}"

        
print("Done.")