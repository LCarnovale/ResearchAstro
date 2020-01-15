import numpy as np
import matplotlib.pyplot as plt

# from HR_fitter import contour_1d

from sys import argv
# if 'big' in argv:
mass_key = 'M'
# else:
#     mass_key = 'star_mass'


valid_vary = [mass_key, 'Z', 'Y', 'alpha', 'diff', 'over'][:-2] # diff and over dont work yet

if len(argv) > 1:
    varying = argv[1]
    no_plot = 'no_plot' in argv
    run_all = False
    if varying == 'all':
        varying = valid_vary[0]
        no_plot = True
        run_all = True
    elif varying not in valid_vary:
        print("Invalid vary paramter given.")
        print("Must be one of:", valid_vary)
        exit()
else:
    print("Give a paramter to vary from:")
    print(valid_vary)
    exit()


import split_grid as sg

def get_colour(v):
    """ Get an RGB colour from a smooth continuous spectrum
    given a value `v` between 0 and 1.
    """
    v %= 1
    RGB = np.array([ # this will cover the whole RGB spectrum
        [1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ])
    RGB = RGB[:, [0, 1, 4, 5]] # Select a colour range that looks good
    nVals = len(RGB[0])

    v = (v * nVals) % nVals # value within [0, nVals)
    left = int(v) % nVals
    right = int(v + 1) % nVals
    dec = v % 1 # value within [0, 1)

    left_rgb = RGB[:, left]
    right_rgb = RGB[:, right]

    shift = right_rgb - left_rgb
    return left_rgb + dec * shift




font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 18}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)

# Get unique tracks
sg.init('mass_conv_core', 'center_degeneracy')
unq = sg.get_unique_tracks()

solar_X = 0.7346
solar_Y = 0.2725; range_Y = 0.0005
solar_Z = 0.01858 ; range_Z = 0.0005
# 1 - 0.7346 - 0.2485 = 0.0169

solar_mass = 1.0; range_mass = 0.001
center_alpha = 1.836; range_alpha = 0.05
center_diff  = 1; range_diff  = 0.01
center_over  = 0.001; range_over  = 0.01

ep_shifts = {
    v:0 for v in valid_vary
}

ep_nu_shifts = {
    v:0 for v in valid_vary
}

vary_vals = {v:list() for v in valid_vary}
trackss = None

def zams_ep(track):
    window_width = 100
    # curv_max = np.argmax(track.curv_inv_rad)
    min_T = np.argmin(track.T_ax[0: window_width])
    # min_T += curv_max - window_width
    return min_T

def h_depletion(track):
    try:
        track.center_h1
    except:
        raise Exception("Unable to see center_h1 field in track.")
    else:
        mask = track.center_h1 < 1e-9
        zero = np.flatnonzero(mask)[0]
        return zero

def density_eps(track):
    d_ax = np.log10(track.M) - 3*track.log_R
    # abs_d = abs(density + 2)
    y_ax = np.arange(len(track))
    # plt.plot(d_ax, y_ax)
    # plt.show()
    idx = np.interp(2., -d_ax, y_ax)
    return idx
    # m = density + 2 < 0
    # return np.flatnonzero(m)[0]

quiver_kwargs = {
    'angles':'xy',
    'scale_units':'xy',
    'scale':1
}

ep_labels = None
def run():
    global ep_shifts
    global trackss
    global vary_vals
    global ep_labels

    max_mass = solar_mass + range_mass
    min_mass = solar_mass - range_mass

    max_alpha = center_alpha + range_alpha
    min_alpha = center_alpha - range_alpha

    max_diff = center_diff + range_diff
    min_diff = center_diff - range_diff

    max_over = center_over + range_over
    min_over = center_over - range_over

    max_Y = solar_Y + range_Y
    min_Y = solar_Y - range_Y

    max_feh = solar_Z + range_Z
    min_feh = solar_Z - range_Z

    m_ax = unq[mass_key]
    feh_ax = unq['Z']
    Y_ax = unq['Y']
    alpha_ax = unq['alpha']
    diff_ax = unq['diffusion']
    over_ax = unq['overshoot']

    m_solar_mask = (m_ax > min_mass) & (m_ax < max_mass)
    feh_solar_mask = (feh_ax < max_feh) & (feh_ax > min_feh)
    Y_solar_mask = (Y_ax < max_Y) & (Y_ax > min_Y)
    alpha_center_mask = (alpha_ax > min_alpha) & (alpha_ax < max_alpha)
    diff_center_mask = (diff_ax > min_diff) & (diff_ax < max_diff)
    over_center_mask = (over_ax > min_over) & (over_ax < max_over)

    fixed_masks = {
        mass_key: m_solar_mask,
        'Z': feh_solar_mask,
        'Y': Y_solar_mask,
        'alpha': alpha_center_mask,
        'diff': diff_center_mask,
        'over': over_center_mask
    }

    all_fixed_mask = np.ones(len(m_solar_mask), dtype=bool)
    for m in fixed_masks:
        if m == varying: continue

        all_fixed_mask &= fixed_masks[m]

    any_found = np.any(all_fixed_mask)

    if not any_found:
        print("No tracks found. :(")
        mask_lens = {k:len(x[x==True]) for k, x in fixed_masks.items()}
        print("mask `True` counts:")
        for m, c in mask_lens.items():
            print(f"{m:<6}: {c}")
    else:
        track_ids = unq['id'][all_fixed_mask]
        # sort them:
        sort_mask = np.argsort(unq[varying][all_fixed_mask])
        track_ids = track_ids[sort_mask]
        track_count = len(track_ids)
        print(f"{track_count} tracks found.")

    if not any_found or 'guide' in argv:
        print("Plotting a helpful diagram")
        m_shift = (m_ax - solar_mass) / range_mass
        feh_shift = (feh_ax - solar_Z) / range_Z
        Y_shift = (Y_ax - solar_Y) / range_Y
        alpha_shift = (alpha_ax - center_alpha) / range_alpha
        diff_shift = (diff_ax - center_diff) / range_diff
        over_shift = (over_ax - center_over) / range_over
        plt.plot(m_shift, '.', label=mass_key)
        plt.plot(feh_shift, '.', label='Z')
        plt.plot(Y_shift, '.', label='Y')
        plt.plot(alpha_shift, '.', label='alpha')
        plt.plot(diff_shift, '.', label='diff')
        plt.plot(over_shift, '.', label='over')
        all_points = np.array([feh_shift, Y_shift, alpha_shift, diff_shift, over_shift, m_shift])
        mins = np.min(all_points, axis=0)
        maxs = np.max(all_points, axis=0)
        for m, mn, mx in zip(np.arange(len(m_shift)), mins, maxs):
            plt.plot([m, m], [mn, mx], linewidth=1, color='gray')
        plt.axhline(1)
        plt.axhline(-1)
        if any_found:
            found_idxs = np.flatnonzero(all_fixed_mask)
            plt.plot(found_idxs, 0*found_idxs, 'rx', label="picked tracks")
        plt.ylabel("ranges from center value")
        plt.xlabel("Unique track")
        plt.legend()
        # plt.show()

    if not any_found:
        plt.show()
        exit()

    plt.figure(f"{varying}_var")

        # return np.argmin(abs_d)

    # def degen_ep(track):
    #     return np.argmax(track.center_degeneracy)

    trackset = sg.TrackSet(
        *[ sg.get_track_from_id(id, 'log_R', 'star_age', 'center_h1') for id in track_ids ]
    )

    trackss = trackset

    trackset.add_ep_func(zams_ep)
    trackset.add_ep_func(h_depletion)
    trackset.add_ep_func(density_eps)
    ep_labels = ['ZAMS', 'H depletion', '1% Solar Avg Density']



    # The rough left/right shift of tracks:
    shift = int(np.sign(trackset[1].T_ax[0] - trackset[0].T_ax[0]))

    for i, track in enumerate(trackset):
        last_track = i + 1 == len(trackset)

        R = 10**track.log_R
        T = track.T_ax

        numax_ax = track.M / ((R**2) * np.sqrt(T/5777.)) * 3100
        nupk = max(numax_ax)

        if not last_track:
            eps_this = trackset.get_ep_points(i)     # shape (<number of eps>, 2)
            eps_next = trackset.get_ep_points(i + 1)
            eps_shift = eps_next - eps_this
            ep_shifts[varying] += eps_shift

            for e in range(trackset.num_eps):
                nu_shift = np.zeros(trackset.num_eps)
                nu_shift[e] = (
                    trackset.interp_ep(i, e, nupk - numax_ax) -
                    trackset.interp_ep(i + 1, e, nupk - numax_ax)
                )
                ep_nu_shifts[varying] += nu_shift
                print(ep_nu_shifts)
                if not no_plot:
                    plt.quiver(
                        eps_this[e, 0], eps_this[e, 1],
                        eps_shift[e, 0], eps_shift[e, 1],
                        **quiver_kwargs, color=f"C{e}", width=0.003,
                        label=(ep_labels[e] if i == 0 else None)
                    )

        vary_val = track.__getattribute__(varying)
        vary_vals[varying].append(vary_val)

        s = f"{varying}: {vary_val:.4f}"

        print(s)

        if last_track or i == 0:
            midpoint = int(len(track) * .75)
            data_point = (track.T_ax[midpoint], track.L_ax[midpoint])
            ha_i, ha_f = ['left', 'right'][::shift]
            if i == 0:
                text_point = (data_point[0]-shift*0.005, data_point[1])
                plt.annotate(
                    s, data_point, xytext=text_point,
                    size='small', ha=ha_i)
            else:
                text_point = (data_point[0]+shift*0.005, data_point[1])
                plt.annotate(
                    s, data_point, xytext=text_point,
                    size='small', ha=ha_f)

        if not no_plot:
            age_ax = ((track.star_age) / 15e9)
            # age_ax /= age_ax[-1]
            step = 4
            for k in range(trackset.eps[i][0], len(track.T_ax), step):
                T = track.T_ax[k:k+step+1]
                L = numax_ax[k:k+step+1]
                age = age_ax[k]
                plt.loglog(T, L, '-', color=get_colour(age))

        # h= plt.plot(track.star_age, track.mass_conv_core)

    ep_shifts[varying] /= (len(trackset) - 1)

    if not no_plot:
        plt.title(f"Evolutionary tracks with varying {varying}, colour coded by Age / 10 Gyr")
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.legend(title="Evolutionary stages")
        plt.xlabel(r"$T_{eff} [K]$")
        plt.ylabel(r"$L$ [solar]")
        plt.show()

if __name__ == '__main__':
    if run_all:
        for v in valid_vary:
            varying = v
            run()

        plt.close('all')

        fig = plt.figure('all_vary')
        axes = fig.subplots(ncols=3, nrows=2)

        arrow_mags = {
            'M' : [0.01, 0.01, 0.01],
            'Y' : [0.01, 0.01, 0.01],
            'Z' : [0.0025, 0.0025, 0.001],
            'alpha': [0.2, 0.2, 0.05]
        }

        for var, vals in vary_vals.items():
            vals = np.array(vals)
            vals = vals[1:] - vals[:-1]
            mean = np.mean(vals)
            ep_shifts[var] /= mean
            ep_nu_shifts[var] /= mean

            vary_vals[var] = mean

        for i, ax in enumerate(axes[0]):
            title = ep_labels[i]
            ax.set_title(title)
            ax.set_xlabel(r"$T_{eff} [K]$")
            ax.set_ylabel(r"$L$ [solar]")
            ax.invert_xaxis()

            for k, var in enumerate(valid_vary):
                arr_mag = arrow_mags[var][i]
                (u, v) = ep_shifts[var][i] * arr_mag
                ax.quiver(
                    u, v, **quiver_kwargs, width=0.007, label=f"{var} [+{arr_mag*100}%]",
                    color=f'C{k}')

            ax.legend()
            ax.set_xlim(150, -150)
            ax.set_ylim(-0.2, 0.2)

        for i, ax in enumerate(axes[1]):
            title = ep_labels[i]
            ax.set_title(title)
            ax.set_xlabel(r"$T_{eff} [K]$")
            ax.set_ylabel(r"$\nu_{max}$")
            ax.invert_xaxis()

            for k, var in enumerate(valid_vary):
                arr_mag = arrow_mags[var][i]
                u = ep_shifts[var][i][0] * arr_mag
                v = ep_nu_shifts[var][i] * arr_mag
                ax.quiver(
                    u, v, **quiver_kwargs, width=0.007, label=f"{var} [+{arr_mag*100}%]",
                    color=f'C{k}')


            ax.legend()
            ax.set_xlim(150, -150)
            ax.set_ylim(1500, -1500)


        plt.show()
    else:
        run()
