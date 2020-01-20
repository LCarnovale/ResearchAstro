from os import listdir
import numpy as np


top_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\hist_grids"

out = "hist_grids/alldat.txt"

valid_vary = ['alpha', 'M', 'Y', 'Z']
files = {a:list() for a in valid_vary}
# fnames = []
for d in valid_vary:
    flist = listdir(top_path + "\\" + d)
    for f in flist:
        # if f in fnames: continue
        # fnames.append(f)
        files[d].append(top_path + "\\" + d + "\\" + f)


def load_hist_track(vary_var, index):
    """ Returns `(inits, data)`, a (length 1) record of initial values
    and a (longer) record of data points."""
    if vary_var not in valid_vary:
        raise ValueError("vary_var was", vary_var, "but must be in", valid_vary)

    try:
        file = files[vary_var][index]
        inits = np.genfromtxt(file, skip_header=1, max_rows=1, names=True)
        inits = {k:inits[k] for k in inits.dtype.names}
        extra_inits = file.split('\\')[-1][:-4].split('_') # split the file name

        extra_labels = ["M", "Y", "Z", "alpha", "diff", "settling", "eta", "overshoot"]
        for init in extra_inits:
            label, val = init.split('=')
            inits[label] = float(val)

        solar_Z = 0.01858
        inits['feh'] = np.log10(inits['Z'] / solar_Z)
        data = np.genfromtxt(file, skip_header=5, names=True)
        data = np.sort(data, order='star_age')
    except IndexError:
        return None
    except Exception as e:
        print("Unable to load file:", file)
        print("Error:", e)
        return None
    else:
        return inits, data


def load_hist_all(vary_var, max=100):
    """ Return all tracks for a given vary parameter.

    It is not known immediately how many tracks will be found so it just keeps
    trying to load them until it breaks, or hits `max` tracks. If it does hit max,
    a warning is raised."""
    out = []
    for i in range(max):
        new = load_hist_track(vary_var, i)
        if new == None:
            return out
        else:
            out.append(new)

    raise Warning(f"Max track count reached in load_hist_all for vary_var = {vary_var}")
    return out
