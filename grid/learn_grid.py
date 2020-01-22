import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append('../')
from tracks import *
import pickle
from numpy.lib import recfunctions as rfn

argv = sys.argv

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 18}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)


big_data_path = 'grid_dump.nm'
meta_dump_path = 'meta.gurkin'

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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

def get_track_from_id(id):
    if id not in data_all['id']:
        id = np.unique(data_all['id'])[id]

    rows = data_all[data_all['id'] == id]
    X = np.array(rows[default_input_labels][0].tolist())
    Y = rows[default_output_labels]
    Y = np.array([list(x) for x in Y])
    return X, Y


def eval_regrs(inputs):
    """ Run the given inputs through all random forests to get a fit.
    Must fit all the forests first obviously.

    Returns an array of shape:
        (<num input samples>, <num steps>, <num output labels>)

    For a single dimensional length N input array, with NR regressors,
    and L output labels, the output will have shape (NR, 1, L).
    If the input is 2 dimensional, to predict multiple input samples,
    then the 1 in that shape is replaced with the number of samples,
    or inputs.shape[0].
    """



    if regr == None:
        print("Please generate the list of regressors with fit_init() before calling this.")
        return

    inputs = np.asarray(inputs)
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(1, -1)

    out = np.zeros(shape=(inputs.shape[0], 96, len(default_output_labels)))

    for i, inp in enumerate(inputs):
        print("Predicting:", inp)
        temp = regr.predict(np.array(inp.tolist()).reshape(1, -1))
        temp = temp.reshape(-1, out.shape[-1])
        out[i] = temp[:]
    return out

regr = None
def fit_init():
    global regr
    num_steps = 96
    # regr_list = [
    regr = RandomForestRegressor(
        min_samples_leaf=num_steps, n_estimators=len(default_input_labels)*64,
        warm_start=True, n_jobs=2, bootstrap=True
    )
    # regr = MLPRegressor(
    #     solver='lbfgs',
    # )
    print("Fetching training data...")
    # training_data = [get_step_from_all(i) for i in range(num_steps)]
    full_tracks = data_all['step_id'] == num_steps - 1
    full_tracks = np.flatnonzero(full_tracks)
    full_tracks = full_tracks.reshape(-1, 1)
    steps = np.arange(num_steps)
    steps -= steps[-1]
    steps = np.tile(steps, [len(full_tracks), 1])
    ft = full_tracks + steps # ft is just short for full tracks

    X = data_all[default_input_labels][ft[:, 0]]
    Y = data_all[default_output_labels][ft]
    Y = Y.reshape(X.shape[0], -1)


    # ids = np.unique(data_all['id'])
    # training_data = [get_track_from_id(i) for i in ids]
    # X = np.concatenate([d[0].reshape(1, -1) for d in training_data if len(d[1]) == num_steps])
    # Y = np.concatenate([d[1].reshape(1, -1) for d in training_data if len(d[1]) == num_steps])

    newX = np.zeros(shape=(len(X), len(default_input_labels)))
    for i, l in enumerate(default_input_labels):
        newX[:, i] = X[l]

    newY = np.zeros(shape=(len(Y), len(default_output_labels) * Y.shape[-1]))
    for i, l in enumerate(default_output_labels):
        newY[:, i::len(default_output_labels)] = Y[l]
        # sep = num_steps
        # newY[:, i*sep : (i+1)*sep] = Y[l]

    print("Fitting...")
    regr.fit(newX, newY)
    print("Done.")

    #
    # i = 0
    # # for regr, data in zip(regr_list, training_data):
    # # for data in training_data:
    # for id in ids:
    #     break
    #     if not i%25:
    #         print(f"Fitting track {i}...", end="\r", flush=True)
    #     X, Y = get_track_from_id(id)
    #     if len(Y) != num_steps:
    #         continue
    #     X = X.reshape(1, -1)
    #     Y = Y.reshape(1, -1)
    #     # print(Y[-1][-1])
    #
    #     # print(X)
    #     # print(Y)
    #     regr.fit(X, Y)
    #     i += 1
    #     if i > 500: break
    # print()
    # print("Done.")


if __name__ == '__main__':
    fit_init()
    heavy_in = solar_inputs.copy()
    heavy_in[0] = 1.2
    inp, model = get_track_from_id(k)
    solar_out = eval_regrs([solar_inputs, heavy_in, inp])
    for out in solar_out:
        plt.plot(out[:, 0], out[:, 1])
    plt.plot(model[:, 0], model[:, 1])
    plt.gca().invert_xaxis()
    plt.title("Fitted track for solar like inputs")
    plt.legend(["1 Solar Mass", f"{heavy_in[0]} Solar Mass", f"{inp[0]:.1f} (Predicted)" ,f"{inp[0]:.1f} Solar Mass (Model)"])
    plt.xlabel("T_eff")
    plt.ylabel("L")
    plt.show()
