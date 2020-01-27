import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append('../')
from tracks import *
import pickle
from numpy.lib import recfunctions as rfn

from matplotlib.widgets import Slider, Button, RadioButtons

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
    'diffusion', 'settling', 'overshoot'
]

solar_inputs = np.array([
    1.0, 0.2725, 0.01858, 1.836,
    1., 1., 0., 0., 0., 0., 0.
])[:len(default_input_labels)]

default_output_labels = [
    'Teff', 'L',
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
        min_samples_leaf=1, n_estimators=len(default_input_labels)*128,
        warm_start=True, n_jobs=-1, bootstrap=True,
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


if __name__ == '__main__':
    fit_init()
    fig = plt.figure("Interpolated Tracks")
    plt.subplots_adjust(bottom=0.15 + .05*len(default_input_labels))
    ax = fig.subplots()
    axcolor = 'lightgoldenrodyellow'

    def update(val):
        new_inp = np.array([s.val for s in sliders])
        print("Custom track values:", new_inp.round(3), end='\r', flush=True)
        track = eval_regrs([new_inp])[0]
        l.set_xdata(track[:, 0])
        l.set_ydata(track[:, 1])
        fig.canvas.draw_idle()

    btn_ax = fig.add_axes([0.8, 0.1, 0.06, 0.03])
    btn = Button(btn_ax, 'Set to Real', color=axcolor, hovercolor='0.975')
    def match_real(event):
        for s, v in zip(sliders, real_inp):
            s.set_val(v)

    btn.on_clicked(match_real)

    sliders = []
    for i, l in enumerate(default_input_labels):
        new_ax = fig.add_axes([0.1, 0.1 + 0.05*i, 0.65, 0.03], facecolor=axcolor)
        vinit = solar_inputs[i]
        vmin = np.min(data_all[l])
        vmax = np.max(data_all[l])
        s = Slider(new_ax, l, vmin, vmax, valinit=vinit)
        s.on_changed(update)
        sliders.append(s)

    # s_M = Slider(ax_M, 'Mass', 0.5, 2, valinit=1)
    # s_Y = Slider(ax_Y, 'Y', 0, 0.3, valinit=solar_inputs[1])

    # a0 = solar_inputs[3]
    # s_alpha = Slider(ax_alpha, r'$\alpha$', a0-1, a0+2, valinit=a0)
    inputs = solar_inputs
    track = eval_regrs([solar_inputs])[0]
    l, = ax.plot(track[:, 0], track[:, 1], label='Custom')
    # solar like stars: [3805, 3438, 4609, 3149, 6219, 6818, 2364, 4492, 2967, 1628]
    real_inp, real_out = get_track_from_id(3805)
    ax.loglog(real_out[:, 0], real_out[:, 1], label=fr"Real track")
    ax.invert_xaxis()

    print("Input parameters   :", np.asarray(default_input_labels))
    print("Real track values  :", real_inp.round(3))
    print("Custom track values:", solar_inputs, end='\r', flush=True)


    ax.legend()
    plt.show()






    # heavy_in = solar_inputs.copy()
    # heavy_in[0] = 1.2
    # inp, model = get_track_from_id(k)
    # solar_out = eval_regrs([solar_inputs, heavy_in, inp])
    # for out in solar_out:
    #     plt.plot(out[:, 0], out[:, 1])
    # ax.plot(model[:, 0], model[:, 1])
    # ax.invert_xaxis()
    # ax.set_title("Fitted track for solar like inputs")
    # ax.set_xlabel("T_eff")
    # ax.set_ylabel("L")
    # ax.legend(["1 Solar Mass", f"{heavy_in[0]} Solar Mass", f"{inp[0]:.1f} (Predicted)" ,f"{inp[0]:.1f} Solar Mass (Model)"])
    # plt.show()
