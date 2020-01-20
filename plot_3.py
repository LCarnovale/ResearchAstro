import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from tracks import *
import ep_funcs as epf

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression

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

if len(argv) < 3:
    print(f"Usage: {argv[0]} <grid type> [spread] <varying parameter>")
    exit()

def check_valid_vary(type, given, valid):
    if given not in valid:
        print(f"{given} is not a valid varying parameter for {type} mode.")
        print(f"The vary parameter must be in {valid}.")
        exit()
    else:
        return True

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
    left = np.asarray(v).astype(int) % nVals
    right = np.asarray(v + 1).astype(int) % nVals
    dec = v % 1 # value within [0, 1)

    left_rgb = RGB[:, left]
    right_rgb = RGB[:, right]

    shift = right_rgb - left_rgb
    return left_rgb + dec * shift


if 'hist' in argv:
    ep_count = 3
    def setup_trackset(ts):
        def create_numax_col(track):
            R = 10**track.log_R
            T = 10**track.log_Teff
            M = track.star_mass
            numax_ax = M / ((R**2) * np.sqrt(T/5777.)) * 3100
            return numax_ax.copy()

        def create_deltanu_col(track):
            R = 10**track.log_R
            T = 10**track.log_Teff
            M = track.star_mass
            delta_nu = M**(1/2) * (T/5777.)**3 / (track.luminosity**(3/4)) * 135
            return delta_nu.copy()

        ts.create_cols(create_numax_col, 'nu_max')
        ts.create_cols(create_deltanu_col, 'delta_nu')
        ts.create_cols(lambda x: 10**x.log_Teff, 'Teff')
        # solar_Z = 0.01858
        # ts.create_cols(lambda x: np.log10(x.Z / solar_Z), 'feh')


        ts.set_xlabel('Teff')

        ts.add_ep_func(epf.zams_ep, 'ZAMS')
        ts.add_ep_func(epf.h_depletion, 'TAMS')
        # ts.add_ep_func(epf.density_eps, '1% Solar Avg Density')
        ts.add_ep_func(epf.delta_nu, r'RGB $\Delta \nu = 10 \mu Hz$')



    import read_hist as rh
    varying = argv[-1]
    if 'plots' in argv :
        L_fig = plt.figure("vary_all_hist_L_T")
        L_axes = L_fig.subplots(nrows=2, ncols=2)
        nu_fig = plt.figure("vary_all_hist_nu_T")
        nu_axes = nu_fig.subplots(nrows=2, ncols=2)
        for v_i, varying in enumerate(rh.valid_vary):
            ax_L = L_axes[v_i//2][v_i%2]
            ax_nu = nu_axes[v_i//2][v_i%2]
            print("Loading tracks...")
            tracks = [Track(init, data) for (init, data) in rh.load_hist_all(varying)]
            tracks = sorted(tracks, key=lambda x:x[varying])
            TS = TrackSet(*tracks)
            print("Done.")
            print("Finding EPs...")

            setup_trackset(TS)

            if varying == 'Z': varying = 'feh'

            print("Done.")

            TS.set_ylabel('nu_max')
            all_nu_eps = np.array([TS.get_ep_points(i) for i in range(len(TS))])
            TS.set_ylabel('luminosity')
            all_L_eps = np.array([TS.get_ep_points(i) for i in range(len(TS))])

            nu_ep_shifts = all_nu_eps[1:] - all_nu_eps[:-1]
            L_ep_shifts = all_L_eps[1:] - all_L_eps[:-1]

            print("Plotting...")
            # plot the tracks
            step_s = 5
            shift = int(np.sign(TS[1].Teff[0] - TS[0].Teff[0]))

            for i, track in enumerate(TS):
                s = f"{track[varying]:.04f}"
                if i == 0 or i + 1 == len(TS):
                    midpoint = int(len(track) * .75)
                    L_data_point = (track.Teff[midpoint], track.luminosity[midpoint])
                    nu_data_point = (track.Teff[midpoint], track.nu_max[midpoint])
                    ha_i, ha_f = ['left', 'right'][::shift]
                    if i == 0:
                        L_text_point = (L_data_point[0]-shift*20, L_data_point[1])
                        nu_text_point = (nu_data_point[0]-shift*20, nu_data_point[1])
                        ax_L.annotate(
                            s, L_data_point, xytext=L_text_point,
                            size='small', ha=ha_i
                        )
                        ax_nu.annotate(
                            s, nu_data_point, xytext=nu_text_point,
                            size='small', ha=ha_i
                        )
                    else:
                        L_text_point = (L_data_point[0]+shift*20, L_data_point[1])
                        nu_text_point = (nu_data_point[0]+shift*20, nu_data_point[1])
                        ax_L.annotate(
                            s, L_data_point, xytext=L_text_point,
                            size='small', ha=ha_f
                        )
                        ax_nu.annotate(
                            s, nu_data_point, xytext=nu_text_point,
                            size='small', ha=ha_f
                        )


                age_ax = (track.star_age / 1e10)**1.3

                start = TS.eps[i][0]
                end = len(age_ax)
                for t in range(start, end, step_s):
                    t_seg = track.Teff[t:t+step_s+1]
                    nu_seg = track.nu_max[t:t+step_s+1]
                    L_seg = track.luminosity[t:t+step_s+1]

                    c = get_colour(age_ax[t])
                    ax_L.semilogy(t_seg, L_seg, color=c)
                    ax_nu.semilogy(t_seg, nu_seg, color=c)

            # Plot the ep points

            for i, ep in enumerate(TS.ep_labels):
                X_nu = all_nu_eps[:-1, i, 0]
                Y_nu = all_nu_eps[:-1, i, 1]
                U_nu = nu_ep_shifts[:, i, 0]
                V_nu = nu_ep_shifts[:, i, 1]

                # if varying == 'M':
                #     # The arrows look a bit small on this graph
                #     ax_nu.quiver(X_nu, Y_nu, U_nu, V_nu, label=ep,
                #         **quiver_kwargs, width=0.012, color=f'C{i}')
                # else:
                ax_nu.quiver(X_nu, Y_nu, U_nu, V_nu, label=ep,
                    **quiver_kwargs, width=0.005, color=f'C{i}', zorder=5)

                X_L = all_L_eps[:-1, i, 0]
                Y_L = all_L_eps[:-1, i, 1]
                U_L = L_ep_shifts[:, i, 0]
                V_L = L_ep_shifts[:, i, 1]

                ax_L.quiver(X_L, Y_L, U_L, V_L, label=ep,
                    **quiver_kwargs, width=0.005, color=f'C{i}', zorder=5)
                # if 'spread' in argv:
                #     spread_nu.quiver(0, 0, U_nu, V_nu, label=ep,
                #         **quiver_kwargs, width=0.004, color=f'C{i}')
                #     spread_L.quiver(0, 0, U_L, V_L, label=ep,
                #         **quiver_kwargs, width=0.004, color=f'C{i}')
                #
                #     spread_nu.legend(title='Evolutionary Stages')
                #     spread_L.legend(title='Evolutionary Stages')

            var_shift = np.mean(np.diff([x[varying] for x in TS]))

            ax_L.legend(title='Evolutionary Stages')
            ax_L.invert_xaxis()
            ax_L.set_xlabel(r"$T_{eff}$ [K]")
            ax_L.set_ylabel(r"$L$ [$L_\odot$]")
            ax_L.set_title("Evolutionary track for varying " + varying + rf", $\Delta${varying}={var_shift:.04f}")

            ax_nu.legend(title='Evolutionary Stages')
            ax_nu.invert_xaxis()
            ax_nu.invert_yaxis()
            ax_nu.set_xlabel(r"$T_{eff}$ [K]")
            ax_nu.set_ylabel(r"$\nu_{max}$ [$\mu H z$]")
            ax_nu.set_title("Evolutionary track for varying " + varying + rf", $\Delta${varying}={var_shift:.04f}")
            # if 'spread' in argv:
            #     spread_L.invert_xaxis()
            #     spread_L.set_ylabel(ax_L.get_ylabel())
            #     spread_L.set_xlabel(ax_L.get_xlabel())
            #     spread_L.set_title("Spread of EP shifts")
            #     spread_nu.invert_xaxis()
            #     spread_nu.invert_yaxis()
            #     spread_nu.set_ylabel(ax_nu.get_ylabel())
            #     spread_nu.set_xlabel(ax_nu.get_xlabel())
            #     spread_nu.set_title("Spread of EP shifts")

    else:
        mag_scale = 100
        fig = plt.figure('all_vary_hist')
        axes_L, axes_nu = fig.subplots(nrows=2, ncols=ep_count)

        fixed_steps = np.array([
            0.15,  # alpha
            0.05, # M
            0.02, # Y
            0.05  # feh
        ])

        vary_labels = rh.valid_vary
        for i, var in enumerate(vary_labels):
            tracks = [Track(init, data) for (init, data) in rh.load_hist_all(var)]
            tracks = sorted(tracks, key=lambda x:x[var])
            TS = TrackSet(*tracks)
            setup_trackset(TS)
            if var == 'Z': var = 'feh'
            TS.set_ylabel('luminosity')
            L_eps = np.array([TS.get_ep_points(i) for i in range(len(TS))])
            TS.set_ylabel('nu_max')
            nu_eps = np.array([TS.get_ep_points(i) for i in range(len(TS))])

            var_shift = np.mean(np.diff([x[var] for x in TS]))

            L_shift = L_eps[1:] - L_eps[:-1]
            nu_shift = nu_eps[1:] - nu_eps[:-1]

            L_mean = np.mean(L_shift, axis=0)
            L_mean /= var_shift
            # Divide by corresponding solar luminosity at these stages
            L_mean[:, 1] /= np.array([TS.interp_ep(4, epi, TS[4].luminosity) for epi in range(TS.num_eps)])
            L_norm = np.linalg.norm(L_mean, axis=1).reshape(-1, 1)
            nu_mean = np.mean(nu_shift, axis=0)
            nu_mean /= var_shift
            # Divide by corresponding solar numax at each stage
            nu_mean[:, 1] /= np.array([TS.interp_ep(4, epi, TS[4].nu_max) for epi in range(TS.num_eps)])
            nu_norm = np.linalg.norm(nu_mean, axis=1).reshape(-1, 1)

            # L_mean /= L_norm.reshape(-1, 1) / mag_scale
            # nu_mean /= nu_norm.reshape(-1, 1) / mag_scale
            # Now the mean arrays have the change per unit step
            if 'fixed' in argv:
                L_step = nu_step = fixed_steps[i] * np.ones(L_norm.shape)
            else:
                L_step = mag_scale / L_norm
                nu_step = mag_scale / nu_norm

            for e in range(ep_count):
                ax_L = axes_L[e]
                ax_nu = axes_nu[e]
                U_L, V_L = L_mean[e] * L_step[e]
                U_nu, V_nu = nu_mean[e] * nu_step[e]
                # L_m = L_mag[e]
                # nu_m = nu_mag[e]

                ax_L.quiver(U_L, 100*V_L, **quiver_kwargs, width=0.007-0.0008*i,
                    color=f'C{i}', label=f'{var} [+{L_step[e][0]:.02f}]')
                ax_nu.quiver(U_nu, 100*V_nu, **quiver_kwargs, width=0.007-0.0008*i,
                    color=f'C{i}', label=f'{var} [+{nu_step[e][0]:.02f}]')

            ep_titles = TS.ep_labels
            # ep_titles[1] = 'TAMS'

            for title, ax_L, ax_nu in zip(TS.ep_labels, axes_L, axes_nu):
                fig.suptitle("Effect of Changing Input Parameters on Stellar Evolution Tracks")
                ax_L.set_title(title)
                ax_L.set_xlabel(r'$\Delta T_{eff}$ [K]')
                ax_L.set_ylabel(r'$\Delta L$ (%) [Relative to solar track]')
                ax_L.invert_xaxis()
                ax_L.set_ylim(-25, 25)
                ax_L.set_xlim(200, -200)
                ax_nu.set_title(title)
                ax_nu.set_xlabel(r'$\Delta T_{eff}$ [K]')
                ax_nu.set_ylabel(r'$\Delta \nu_{max}$ (%) [Relative to solar track]')
                ax_nu.invert_yaxis()
                ax_nu.invert_xaxis()
                ax_nu.set_ylim(10, -10)
                ax_nu.set_xlim(200, -200)
                ax_L.legend(title="Input changes")
                ax_nu.legend(title="Input changes")







plt.show()
