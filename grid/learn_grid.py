import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
sys.path.append(r'C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello')
from tracks import Track, TrackSet
from tracks import *
import pickle
from numpy.lib import recfunctions as rfn
import tensorflow as tf
from joblib import load, dump

from matplotlib.widgets import Slider, Button, CheckButtons

argv = sys.argv

model_path = 'model.joblib'

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 18}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

quiver_kwargs = {
    'angles':'xy',
    'scale_units':'xy',
    'scale':1
}



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

input_shifts = [
    0.05, 0.02, -0.05, 0.15,
    0.1, 0.1, 0.1,
]

solar_inputs = np.array([
    1.0, 0.2725, 0.01858, 1.836,
    1., 1., 0., 0., 0., 0., 0.
])[:len(default_input_labels)]

default_output_labels = [
    'Teff', 'L', 'X_c', 'nu_max', 'age',
    # M, Y, Z, alpha, diffusion, settling, eta, overshoot, undershoot,
    # overexp, underexp, age, M_current, mass_cc, h_exh_core_mass,
    # h_exh_core_radius, radius_cc, cz_mass, cz_radius,
    # mass_X, mass_Y, log_LH, log_LHe, log_center_T, log_center_Rho,
    # log_center_P, center_mu, center_degeneracy, Teff, log_Teff, L,
    # log_L, radius, log_R, log_g, Fe_H, acoustic_cutoff,
    # acoustic_radius, CCB_radius, CCB_mass, CCB_Hp, f0_radius,
    # f0_mass, Eff_Hp, hp_step, FM_radius, FM_mass, OS_region_R,
    # OS_region_M, Eff_OS_scale, surface_mu, delta_Pg_asym, nu_max,
    # delta_nu_asym, X_c, Y_c, Li_c, Be_c, B_c, C_c, N_c, O_c, F_c, Ne_c,
    # Mg_c, X_surf, Y_surf, Li_surf, Be_surf, B_surf, C_surf, N_surf,
    # O_surf, F_surf, Ne_surf, Mg_surf, Dnu0, dnu02,
    # r02, r01, dnu13, r13, r10
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


def eval_regrs(inputs, ts=None):
    """ Run the given inputs through all random forests to get a fit.
    Must fit all the forests first obviously.

    Returns an array of shape:
        (<num input samples>, <num steps>, <num output labels>)

    For a single dimensional length N input array, with NR regressors,
    and L output labels, the output will have shape (NR, 1, L).
    If the input is 2 dimensional, to predict multiple input samples,
    then the 1 in that shape is replaced with the number of samples,
    or inputs.shape[0].

    If a trackset is given with `ts`, the new tracks will be added to it,
    and the new tracks` indexes returned instead (or whatever ts.add_track()
    returns for each track.) as a list.
    """



    if regr == None:
        print("Please generate the list of regressors with fit_init() before calling this.")
        return

    inputs = np.asarray(inputs)
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(1, -1)

    outs = np.zeros(shape=(inputs.shape[0], 96, len(default_output_labels)))

    for i, inp in enumerate(inputs):
        temp = regr.predict(np.array(inp.tolist()).reshape(1, -1))
        temp = temp.reshape(-1, outs.shape[-1])
        outs[i] = temp[:]

    track_init = [{k:input[i] for (i, k) in enumerate(default_input_labels)} for input in inputs]
    track_data = [{k:out[:, d] for (d, k) in enumerate(default_output_labels)} for out in outs]
    tracks_out = [Track(ti, td) for (ti, td) in zip(track_init, track_data)]

    if ts is not None:
        return [ts.add_track(t) for t in tracks_out]
    else:
        return tracks_out



regr = None
def fit_init():
    global regr

    num_steps = 96
    regr = RandomForestRegressor(
        min_samples_leaf=1, n_estimators=len(default_input_labels)*64,
        warm_start=True, n_jobs=-1, bootstrap=True,
    )
    print("Fetching training data...")
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
    # newX = tf.constant(newX)
    # newY = tf.constant(newY)
    if 'small' in argv:
        regr.fit(newX[:200], newY[:200])
    else:
        regr.fit(newX, newY)
    print("Done.")
    # print("Attempting to save model...")
    # try:
    #     dump(regr, model_path)
    # except Exception as e:
    #     print("Unable to save model. Error:")
    #     print(e)
    # else:
    #     print("Model saved")

import ep_funcs

def tams(track):
    # idx = default_output_labels.index('X_c')
    mask = track.X_c < 1e-4
    zero = np.flatnonzero(mask)
    if len(zero):
        zero = zero[0]
    else:
        zero = -1
        print("Warning: Couldn't find TAMS point")
    # return 60
    return zero

def zams(track):
    curv, tangent, dists = ep_funcs.get_curvature(track, 'Teff', 'L')
    # print(curv[1:15])
    return 1
    # return np.argmax(curv[1:8]) + 1



if __name__ == '__main__':
    fit_init()
    ts = TrackSet(xlabel='Teff', ylabel='L')
    # TS.set_xlabel('Teff')
    # TS.set_ylabel('L')
    ts.add_ep_func(tams, label='tams')
    ts.add_ep_func(zams, label='zams')
    # ts.create_cols(set_curvature_dist,
    #     ["curv_inv_rad", "curv_norm", "tangent", "path_len"]
    # )

    fig = plt.figure("Interpolated Tracks")
    plt.subplots_adjust(bottom=0.15 + .05*len(default_input_labels))
    ax, ax2 = fig.subplots(ncols=2)
    axcolor = 'lightgoldenrodyellow'

    L_scale_unit = 20
    T_scale_unit = 100

    ax.set_title("Predicted Evolutionary Track")

    ax2.set_xlabel("Age (Gyr)")
    arrows_fig = plt.figure("EP shifts")
    plt.subplots_adjust(bottom=0.25)
    arrows_ax_all = arrows_fig.subplots(nrows=ts.num_eps)
    load_button_ax = arrows_fig.add_axes([0.25, 0.05, 0.5, 0.05])
    load_btn = Button(load_button_ax, 'Calculate Shifts', color=axcolor, hovercolor='0.975')
    live_chkbox_ax = arrows_fig.add_axes([0.8, 0.05, 0.05, 0.05])
    live_chkbox = CheckButtons(live_chkbox_ax, ['Live Updating'], [False])
    arrows_all = []
    for a, arrows_ax in enumerate(arrows_ax_all):
        arrows_ax.invert_xaxis()
        arrows_ax.set_title(ts.ep_labels[a])
        arrows_ax.set_xlabel(r"$\Delta T_{eff}$ [K]")
        arrows_ax.set_ylabel(r"$\Delta L$ [%] (Relative to centre track)")
        arrows_ax.set_xlim(50, -50)
        arrows_ax.set_ylim(-20, 20)

        arrows = []
        for i, (l, s) in enumerate(zip(default_input_labels, input_shifts)):
            if s > 0:
                lab = f"{l}+{s}"
            else:
                lab = f"{l}+{-s*1e2}%"

            h = arrows_ax.quiver(1., 1., **quiver_kwargs, width=0.005,
                label=lab, color=f"C{i}")
            arrows.append(h)
        arrows_ax.legend()
        arrows_all.append(arrows)

    def update_arrows(val=None):
        ts.delete_all_tracks()
        new_inp = np.array([s.val for s in sliders])
        center_track = eval_regrs([new_inp], ts=ts)[0]
        center_eps = ts.get_ep_points(center_track)
        center_L = center_eps[:, 1]

        ident = np.identity(len(new_inp))
        inp_shifts = np.array(input_shifts[:])
        inp_shifts[inp_shifts < 0] *= -new_inp[inp_shifts < 0]
        shifts = ident * inp_shifts
        shifted_inps = shifts + new_inp
        # print(shifted_inps)
        outs = eval_regrs(shifted_inps, ts=ts)

        for i in range(len(default_input_labels)):
            inp = shifted_inps[i]
            track = ts[outs[i]]
            # track_tams = ts.tams[outs[i]]
            track_eps = ts.get_ep_points(outs[i])
            dT = (track_eps - center_eps)[:, 0]
            dL = (track_eps - center_eps)[:, 1]
            dL /= center_L
            dL *= 100 # Convert to a percentage of the centre track
            # print(dT, dL)
            for a, ax in enumerate(arrows_all):
                arrow_len = ((dT[a]/T_scale_unit)**2 + (dL[a]/L_scale_unit)**2)**(1/2)

                scale_factor = 0
                if (arrow_len > 1e-6).any():
                    while (arrow_len < 0.1).any():
                        arrow_len *= 10
                        scale_factor += 1
                        # print("arrow length:", arrow_len)
                arrow_h = ax[i]
                arrow_h.U[0] = dT[a] * 10**scale_factor
                arrow_h.V[0] = dL[a] * 10**scale_factor
                base_label = arrow_h.get_label()
                base_label = base_label.split(" $")[0]
                if scale_factor >= 1:
                    scale_factor = "{%d}"%scale_factor
                    new_label = rf"{base_label} $\times 10^{scale_factor}$"
                else:
                    new_label = base_label
                arrow_h.set_label(new_label)

                arrows_ax_all[a].legend()
        arrows_fig.canvas.draw_idle()

    load_btn.on_clicked(update_arrows)

    def update(val):
        ts.delete_all_tracks()
        new_inp = np.array([s.val for s in sliders])
        tracki = eval_regrs([new_inp], ts=ts)[0]
        print("EP indexes : ", ts.eps[0], end='\r', flush=True)
        track = ts[tracki]
        l.set_xdata(track.Teff)
        l.set_ydata(track.L)
        tams_x, tams_y = ts.get_ep_points(tracki).T
        tams_l.set_xdata(tams_x)
        tams_l.set_ydata(tams_y)
        age_ax = track['age']
        keys = default_output_labels[2:-1]
        for i, el in enumerate(extra_curves):
            el.set_ydata(track[keys[i]])
            el.set_xdata(age_ax)
        ax2.set_xlim(-0.1*np.max(age_ax), 1.1*np.max(age_ax))

        # rescale:
        xmin = np.min([track.Teff, real_out[:, 0]])
        xmax = np.max([track.Teff, real_out[:, 0]])
        ymin = np.min([track.L, real_out[:, 1]])
        ymax = np.max([track.L, real_out[:, 1]])
        xrange = xmax - xmin
        yrange = ymax - ymin
        ax.set_xlim(xmax + 0.1*xrange, xmin - 0.1*xrange)
        ax.set_ylim(ymin*0.95, ymax*1.05)
        if live_chkbox.get_status()[0]:
            update_arrows()
        else:
            fig.canvas.draw_idle()
        # ax2.autoscale(axis='y')

    btn_ax = fig.add_axes([0.8, 0.1, 0.1, 0.03])
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
        step = input_shifts[i]
        if step > 0:
            for x in np.arange(vmin, vmax, step=step):
                new_ax.axvline(x)
        else:
            step *= -1
            ratio = 1/(1+step)
            powers = np.arange(0, 50)
            for x in (vmax * ratio**powers):
                new_ax.axvline(x)
        s = Slider(new_ax, l, vmin, vmax, valinit=vinit)
        s.on_changed(update)
        sliders.append(s)

    inputs = solar_inputs
    tracki = eval_regrs([solar_inputs], ts=ts)[0]
    track = ts[tracki]
    extra_curves = []
    for i, l in enumerate(default_output_labels[2:-1]):
        h, = ax2.plot(track['age'], track[l], label=l)
        extra_curves.append(h)

    l, = ax.plot(track.Teff, track.L, label='Custom')
    tams_point = ts.get_ep_points(tracki)
    tams_l, = ax.plot(tams_point[:,0], tams_point[:,1], 'ro', label='TAMS')
    # solar like stars: [3805, 3438, 4609, 3149, 6219, 6818, 2364, 4492, 2967, 1628]
    real_inp, real_out = get_track_from_id(3805)
    ax.semilogy(real_out[:, 0], real_out[:, 1], label=fr"Real track")
    ax.invert_xaxis()
    print("Input parameters   :", np.asarray(default_input_labels))
    print("Real track values  :", real_inp.round(3))

    print("EP labels  : ", np.asarray(ts.ep_labels))
    print("EP indexes : ", ts.eps[0], end='\r', flush=True)


    ax.set_xlabel("T_eff")
    ax.set_ylabel("L")
    update_arrows(0)
    ax.legend()
    ax2.legend()
    arrows_ax.legend()


    plt.show()
    print()






    # heavy_in = solar_inputs.copy()
    # heavy_in[0] = 1.2
    # inp, model = get_track_from_id(k)
    # solar_out = eval_regrs([solar_inputs, heavy_in, inp])
    # for out in solar_out:
    #     plt.plot(out[:, 0], out[:, 1])
    # ax.plot(model[:, 0], model[:, 1])
    # ax.invert_xaxis()
    # ax.set_title("Fitted track for solar like inputs")
    # ax.legend(["1 Solar Mass", f"{heavy_in[0]} Solar Mass", f"{inp[0]:.1f} (Predicted)" ,f"{inp[0]:.1f} Solar Mass (Model)"])
    # plt.show()
