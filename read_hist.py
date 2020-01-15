from os import listdir
import numpy as np
from numpy.lib import recfunctions as rfn


top_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\hist_grids"

out = "hist_grids/alldat.txt"

dirs = ['alpha', 'M', 'Y', 'Z']
files = []
fnames = []
for d in dirs:
    flist = listdir(top_path + "\\" + d)
    for f in flist:
        if f in fnames: continue
        fnames.append(f)
        files.append(top_path + "\\" + d + "\\" + f)

keep_headers = [
    'star_age',# '<f8'),
    'star_mass',# '<f8'),
    'log_R',# '<f8'),
    'log_Teff',# '<f8'),
    'log_L',# '<f8')
    'center_h1',
]

init_conds_headers = [
    'M', 'Y', 'Z', 'alpha', 'diffusion', 'eta', 'overshoot',
]



def get_full_cols(*additional_columns):
    # global headers_dtypes
    global keep_headers
    global init_conds_headers
    keep_headers += additional_columns
    headers_dtypes = [(x, '<f8') for x in keep_headers]
    headers_dtypes += [(x, '<f8') for x in init_conds_headers]
    init_conds_headers += ['id']# just to add the id column
    headers_dtypes += [('id', 'i4')]

    fullcols = np.zeros(0, dtype=headers_dtypes)
    i = 1
    keys = keep_headers + init_conds_headers
    for file in files:
        init_conds = file.split("\\")[-1][:-4]
        init_conds = init_conds.split("_")
        vals = [float(x.split("=")[1]) for x in init_conds]
        M, Y, Z, alpha, diff, settling, eta, over = vals
        data = np.genfromtxt(file, skip_header=5, names=True)
        # sort the arrays
        data = np.sort(data, order='model_number')
        data = rfn.append_fields(data, init_conds_headers,
            [0. for _ in init_conds_headers][:-1] + [1],
            usemask=False)

        data['id'] = i
        data['M'] = M
        data['Y'] = Y
        data['Z'] = Z
        data['alpha'] = alpha
        data['diffusion'] = diff
        data['eta'] = eta
        data['overshoot'] = over

        fullcols = np.append(fullcols[keys], data[keys])
        i += 1

    # header = ','.join(keys)
    return fullcols
