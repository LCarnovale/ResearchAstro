import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append('../')
from tracks import *
import pickle
from numpy.lib import recfunctions as rfn

argv = sys.argv

big_data_path = 'grid_dump.nm'
meta_dump_path = 'meta.gurkin'

from sklearn.ensemble import RandomForestRegressor

files = [
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_0.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_1.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_2.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_3.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_4.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_5.txt",
    r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\grid\out_split_6.txt",
]

default_input_labels = [
    'M', 'Y', 'Z', 'alpha',
    'diffusion', 'settling'
]

solar_inputs = np.array([
    1.0, 0.2725, 0.01858, 1.836,
    1., 1., 0., 0., 0., 0., 0.
])[:len(default_input_labels)]

default_output_labels = [
    'Teff', 'L', 'radius', 'nu_max',
    # 'age', 'mass_cc', 'h_exh_core_mass', 'h_exh_core_radius',
    # 'radius_cc', 'mass_X', 'mass_Y', 'log_LH', 'log_LHe',
    # 'log_center_T', 'log_center_Rho', 'log_center_P', 'center_mu',
    # 'center_degeneracy', 'Teff', 'log_Teff', 'L', 'log_L', 'radius',
    # 'log_R', 'log_g', 'surface_mu', 'nu_max', 'delta_nu_asym',
    # 'X_c', 'Y_c', 'Dnu0'
]

data_all = None

def write_splits():
    # The first one has the names
    grid = np.genfromtxt(files[0], delimiter=',', names=True)

    print("1 / %d"%len(files))
    try:
        for i, f in enumerate(files[1:]):
            next = np.genfromtxt(f, delimiter=',', names=grid.dtype.names)
            grid = np.append(grid, next)
            print("%d / %d"%(i+2, len(files)));
    except KeyboardInterrupt:
        print("Stopping.")

    grid = np.sort(grid, order=('id', 'age'))
    step_id_ax = np.arange(len(grid))
    unique_models = np.unique(grid['id'], return_index=True)[1]


    for i in unique_models:
        step_id_ax[i:] -= step_id_ax[i]

    grid = rfn.append_fields(grid, 'step_id', step_id_ax)

    with open(meta_dump_path, 'wb') as fl:
        pickle.dump([grid.dtype, grid.shape], fl)

    fp = np.memmap(big_data_path, dtype=grid.dtype, shape=grid.shape, mode='w+')
    fp[:] = grid[:].copy()

    del fp

def load_splits():
    with open(meta_dump_path, 'rb') as fl:
        dtype, shape = pickle.load(fl)

    fp = np.memmap(big_data_path, dtype=dtype, shape=shape, mode='r')
    return fp

try:
    if 'rewrite' in argv: raise FileNotFoundError()
    f = open(meta_dump_path, 'r')
    f.close()
    f = open(big_data_path, 'r')
except FileNotFoundError:
    print("Loading data...")
    write_splits()
except KeyboardInterrupt:
    print("Stopping.")
else:
    f.close()
finally:
    data_all = load_splits()
    print("Ready.")

def get_ids(data=None):
    if data is None:
        data = data_all

    return np.unique(data['id'])


def get_Track_from_id(id, data=None):
    if data is None:
        data = data_all

    track_data = data[data['id'] == id]
    track_data = np.sort(track_data, order='age')

    inits = []
    data = []

    for k in keys(track_data):
        if abs(np.std(track_data[k])) > 1e-12:
            data.append(k)
        else:
            inits.append(k)

    return Track(track_data[0][inits], track_data[data])

def get_step_from_all(step, data=None):
    """ Return default input and output (as a tuple (X, Y) ) for a given step.
    For the default big grid step must be between 0 and 95.

    """
    if data is None:
        data = data_all


    rows = data[data['step_id'] == step]

    X = np.array(rows[default_input_labels].tolist())
    Y = np.array(rows[default_output_labels].tolist())
    return X, Y


def eval_regrs(inputs):
    """ Run the given inputs through all random forests to get a fit.
    Must fit all the forests first obviously.

    Returns an array of shape:
        (<num steps/regressors>, <input samples>, <num output labels>)

    For a single dimensional length N input array, with NR regressors,
    and L output labels, the output will have shape (NR, 1, L).
    If the input is 2 dimensional, to predict multiple input samples,
    then the 1 in that shape is replaced with the number of samples,
    or inputs.shape[0].
    """



    if regr_list == None:
        print("Please generate the list of regressors with fit_init() before calling this.")
        return

    inputs = np.asarray(inputs)
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(1, -1)

    out = np.zeros(shape=(len(regr_list), inputs.shape[0], len(default_output_labels)))
    for i, r in enumerate(regr_list):
        out[i] = r.predict(np.array(inputs.tolist()))

    return out

regr_list = None
def fit_init():
    global regr_list
    num_steps = 96
    regr_list = [
        RandomForestRegressor(
                min_samples_leaf=50, n_estimators=200, warm_start=True,
                n_jobs=2,
            ) for i in range(num_steps)
    ]
    print("Fetching training data...")
    training_data = [get_step_from_all(i) for i in range(num_steps)]
    # X = np.concatenate([d[0] for d in training_data])
    # Y = np.concatenate([d[1] for d in training_data])
    print("Fitting...")
    # regr.fit(X, Y)
    # print("Done.")

    #
    i = 0
    for regr, data in zip(regr_list, training_data):
    # for data in training_data:
        print(f"Training step {i}...", end="\r", flush=True)
        X, Y = data
        # print(X)
        # print(Y)
        regr.fit(X, Y)
        i += 1
    print()
    print("Done.")


if __name__ == '__main__':
    fit_init()
    heavy_in = solar_inputs.copy()
    heavy_in[0] = 1.3
    solar_out = eval_regrs([solar_inputs, heavy_in])
    plt.plot(solar_out[:, 0, 0], solar_out[:, :, 1])
    plt.gca().invert_xaxis()
    plt.title("Fitted track for solar like inputs")
    plt.legend(["1 Solar Mass", "1.3 Solar Mass"])
    plt.xlabel("T_eff")
    plt.ylabel("L")
    plt.show()
