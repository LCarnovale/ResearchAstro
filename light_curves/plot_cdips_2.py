import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
import lightkurve as lk
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle
from sys import argv
import sys
from astroquery.mast import Observations
from astroquery.mast import Catalogs

table_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\MAST_Crossmatch_TIC_no_Nans_cs.csv"
time_name = "TMID_BJD" # Exp mid-time in BJD_TDB (BJDCORR applied)

# 126275079 140538401 126544393 126273786 140330119 140538877 126781167 140539089 140538926 140539108 126544385 140538994

table_data = np.genfromtxt(table_path, delimiter=',', names=True, comments='#', skip_header=4)

def gen_dist_mat(coords):
    """ For 2d input array of shape (N, d),
    Return a difference matrix of shape (N, N, d) and
    a distance matrix of shape (N, N).
    """
    N = len(coords)
    pos_all = np.tile(coords, (N, 1, 1))
    pos_s   = np.tile(coords, (1, 1, N)).reshape(pos_all.shape)

    diff = pos_all - pos_s
    dist = np.linalg.norm(diff, 2, axis=-1)

    return diff, dist

cluster_coords = ['ra', 'dec', 'plx']
if 'cluster' in argv:
    max_dist = 2
    if len(argv) < 3:
        print("Give an index to select a cluster.")
    print("Searching for clusters by:", cluster_coords)
    coords = np.array([list(x) for x in table_data[cluster_coords]])
    diff, dist = gen_dist_mat(coords)
    try:
        max_dist = float(argv[3])
    except:
        print("Using default max distance of", max_dist)

    dist_sort = np.argsort(dist, axis=1) # Sorts each star by closest to furthest,
                                         # for each star. (N x N array)
    clust_mems = dist < max_dist
    clust_mem_counts = np.count_nonzero(clust_mems, axis=1)
    dist_sort_list = [row[:count] for (row, count) in zip(dist_sort, clust_mem_counts)]






font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 17}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)
plt.rc('lines', markersize=10)

# Line segment points:
lsx1, lsy1 = 1, 2
lsx2, lsy2 = 2, -2
dist_limit = 0.31

def get_mask(table):
    # Forst sort by decreasing magnitude
    plx = table['plx'] / 1e3
    M = table['GAIAmag'] + 5*np.log10(plx) + 5
    # M_sort = np.argsort(-M)

    bprp = table['gaiabp'] - table['gaiarp']

    d, t = shortest_distance_to_lineseg(
        bprp, M, lsx1, lsy1, lsx2, lsy2
    )
    t_sort = np.argsort(table['Tmag'])


    # inside = table[d < dist_limit]

    return t_sort[d[t_sort] < dist_limit]

def shortest_distance_to_lineseg(xdata, ydata, vx1, vy1, vx2, vy2):
    """ Return the shortest distance from all points made by (xdata, ydata)
    to the line segment joining (vx1, vy1) and (vx2, vy2).

    xdata/ydata: Arraylike, one dimensional.
    vXX : single scalar.

    Return an array of distances, and an array of the component of all
    points along the given line. Points closest to the beginning of the segment
    will have values 0 in this array, points closest to the end 1, and
    the trend is continued linearly for all points along the line.

    Math taken from:
        https://math.stackexchange.com/questions/2248617
    """
    d = np.zeros(xdata.size)
    # If we parameterize the line segment by the single variable t from 0 -> 1,
    # then we can find the value of t which corresponds to the point on the line
    # which is closest to our point of interest:
    t = -((vx1-xdata)*(vx2-vx1)+(vy1-ydata)*(vy2-vy1))                         \
       / ((vx2-vx1)**2+(vy2-vy1)**2)
    # For points where t is outside [0, 1], the closest point on the line is
    # outside of the segement, so we use the minimum of the straight line
    # distance to either end of the segment.
    # Points where t is within [0, 1], we can use the perpendicular distance
    # to the line:
    pm = np.logical_and(t >= 0, t <= 1) # perpendicular mask
    d[pm] = np.abs((vx2-vx1)*(vy1-ydata[pm]) - (vy2-vy1)*(vx1-xdata[pm]))      \
          / np.sqrt((vx2-vx1)**2 + (vy2-vy1)**2)

    # Distances to endpoints:
    d1 = np.sqrt((xdata-vx1)**2 + (ydata-vy1)**2).reshape(-1, 1)
    d2 = np.sqrt((xdata-vx2)**2 + (ydata-vy2)**2).reshape(-1, 1)

    d_min = np.min(np.concatenate([d1, d2], axis=1), axis=1)
    d[~pm] = d_min[~pm]

    return d, t

index_provided = False
def get_cdips_product(star_num):
    """ Given a list of numbers or a single number,
    return the same number of FitsTable objects.

    Given values can be TIC ids or indexes, the latter will be used to index

    """
    global index_provided
    if type(star_num) != list:
        star_num = list(star_num)

    num_ints = [int(n) for n in star_num]
    tic_searches = []
    mask = get_mask(table_data)
    print("Filtered down to", np.count_nonzero(mask), "items to index from.")
    temp_table = table_data[mask]
    num_stars = len(table_data)
    for n in num_ints:
        if n > num_stars:
            # Already have tics
            tic_searches.append(str(n))
        else:
            if not index_provided: index_provided = True
            tic_searches.append(str(int(temp_table['MatchID'][n])))

    obs = Observations.query_criteria(target_name=tic_searches, provenance_name="CDIPS")
    print(f"TIC {tic_searches} returned {len(obs)} object(s).")
    # Get rid of duplicates across sectors
    u, u_indexes = np.unique(obs['target_name'], return_index=True)
    if len(u) != len(obs):
        obs = obs[u_indexes]
        print(f"After removing duplicates, {len(obs)} object(s) remain.")

    if len(obs) > 0:
        products = Observations.get_product_list(obs)
        print("Downloading products...")
        down = Observations.download_products(products)
        print("Done.")
        ft_list = []
        for f in down:
            ft = FitsTable(f['Local Path'])
            ft_list.append(ft)

        ft_list = sorted(ft_list, key=lambda x: tic_searches.index(x.ticid))

        return ft_list
    else:
        return

def get_fits_from_path(path):
    try:
        hdul = fits.open(path)
    except FileNotFoundError:
        print("fits file not found:")
        print(path)
        print("Arguments:")
        print("Path:", path)
    else:
        return hdul


def sigma_clip(data, max_sigma):
    """ Return a mask of values within the given sigma range """
    mn = np.mean(data)
    std = np.std(data)
    diff = data - mn
    sigmas = diff / std
    mask = np.abs(sigmas) < max_sigma
    return mask


class FitsTable:
    def __init__(self, path):
        self._path = path
        self._hdul = get_fits_from_path(path)
        self.gaiaid = self._hdul[0].header['Gaia-ID']
        self._data = self._hdul[1].data
        self._open = True

        self._pca_lcs = {}
        self._tfa_lcs = {}


    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"FitsTable(...{self._path[-40:]})"

    def __getattr__(self, attr):
        try:
            row = table_data[table_data['MatchID'] == int(self.ticid)][0]
            return row[attr]
        except:
            raise AttributeError("Unkown/Invalid attribute: " + attr)

    def close(self):
        if self._open:
            self._hdul.close()
            self._open = False
        else:
            print(repr(self), 'already closed.')

    def open(self):
        if not self._open:
            self._hdul = get_fits(self.sector, self.cam, self.ccd, self.gaiaid)
            self._open = True
        else:
            print(repr(self), 'already open.')

    def get_mag(self, source_type, ap=1, clip=None):
        # ap (aperture) can be 1, 2 or 3
        sc = source_type.upper()
        if sc not in ['PCA', 'TFA']:
            raise Exception("source type must be either tfa or pca")

        mag = self._data[f'{sc}{ap}'].copy()
        if clip:
            msk = sigma_clip(mag, clip)
            mag = mag[msk]

        return mag

    def get_time_ax(self, zero_start=True):
        """ Get the time axis for magnitude measurements.
        By default the start value is set to zero,
        set `zero_start=False` to avoid this.

        Units are probably in days."""
        t_ax = self._data[time_name].copy()
        if zero_start:
            t_ax -= t_ax[0]

        return t_ax

    def get_lc(self, source_type, ap=1, clip=None, force_recalc=False):
        """ `source_type` should be 'tfa' or 'pca'.
        `ap` should be 1 (default), 2 or 3.
        if `clip` is not `None`, then values further than `clip` sigmas
        are removed. """

        if source_type == 'pca':
            lcs = self._pca_lcs
        else:
            lcs = self._tfa_lcs

        if ap in lcs and not force_recalc:
            lc = lcs[ap]
        else:
            flux = self.get_mag(source_type, ap)
            nan_mask = np.isnan(flux)
            flux = flux[~nan_mask]
            t_ax = self.get_time_ax()[~nan_mask]

            if clip:
                mask = sigma_clip(flux, clip)
                flux = flux[mask]
                t_ax = t_ax[mask]
            # flux = 10**(-0.4*flux)
            lc = lk.LightCurve(time=t_ax, flux=flux)
            lcs[ap] = lc
        return lc

    def get_ft(self, source_type, ap=1, clip=None, force_recalc=False):
        lc = self.get_lc(source_type, ap, clip, force_recalc)
        lc.flux *= -1
        pg = lc.to_periodogram(freq_unit=u.microhertz)
        freq = pg.frequency
        power = pg.power
        return (freq, power)

    def query_mast(self):
        """ Fetch the data for this star in astroquery.mast using the TICID
        """
        t = Catalogs.query_criteria(catalog='TIC', target_name=self.ticid)
        return t


    @property
    def cdipsref(self):
        return self._hdul[0].header['CDIPSREF']

    @property
    def cluster(self):
        return self._hdul[0].header['CDCLSTER']

    @property
    def ticid(self):
        return self._hdul[0].header['TICID']

    @property
    def ra(self):
        return self._hdul[0].header['BTC_RA'] # Right ascen in barycentric time correction

    @property
    def dec(self):
        return self._hdul[0].header['BTC_DEC'] # Declination in barycentric time correction

    @property
    def TESSmag(self):
        return self._hdul[0].header['TESSMAG']

    @property
    def plx(self):
        return self._hdul[0].header['Parallax[mas]']

    @property
    def gmag(self):
        return self._hdul[0].header['phot_g_mean_mag'] # assuming this is the colour index we're looking for

    @property
    def bmag(self):
        return self._hdul[0].header['phot_bp_mean_mag'] # assuming this is the colour index we're looking for

    @property
    def rmag(self):
        return self._hdul[0].header['phot_rp_mean_mag'] # assuming this is the colour index we're looking for

    @property
    def Teff(self):
        return self._hdul[0].header['teff_val']

    @property
    def lum(self):
        return self._hdul[0].header['lum_val']

# class InteractivePlotHandler:
#     def __init__(self, fig):




if __name__ == '__main__':
    # plt.ion()
    ids = []
    for s in argv:
        try:
            i = int(s)
            ids.append(i)
        except:
            continue

    if not ids:
        print("Provide ids to search for CDIPS products.")
        exit()
    else:
        fts = get_cdips_product(ids)

    if not fts:
        print("None found.")
        exit()

    nan_mask = ~np.isnan(table_data['plx'])
    bprp = table_data['gaiabp'] - table_data['gaiarp']
    bprp = bprp[nan_mask]
    plx = table_data['plx'][nan_mask] / 1e3
    M = table_data['GAIAmag'][nan_mask] + 5*np.log10(plx) + 5


    fig = plt.figure(f"All plots")
    ncurves = len(fts)
    nrows = ((ncurves-1)//5 + 1)
    ncols = (ncurves + nrows//2) // nrows
    axes = fig.subplots(nrows=2*nrows, ncols=ncols)
    if ncurves == 1:
        axes = axes.reshape(-1, 1)
    pg_axes = axes[::2, :].flatten()
    hr_axes = axes[1::2, :].flatten()

    row_i = 0
    col_i = 0
    # if 'cluster' in argv or True:
    bprp_selection = np.array([f.bmag - f.rmag for f in fts])
    M_selecion = np.array([f.gmag+5*np.log10(f.plx/1e3)+5 for f in fts])
    sub_mask = get_mask(table_data)

    # For clusters:
    sort = np.argsort([f.bmag-f.rmag for f in fts])
    fts = [fts[i] for i in sort]
    M_selecion = M_selecion[sort]
    bprp_selection = bprp_selection[sort]

    # chckbox_ax = fig.add_axes([0.1, 0.95, 0.02, 0.02])



    while len(fts) < len(pg_axes):
        fts.append(None)

    for i, (pg_ax, hr_ax, f) in enumerate(zip(pg_axes, hr_axes, fts)):
        if f is None:
            fig.delaxes(pg_ax)
            fig.delaxes(hr_ax)
            continue
        print(f"Plotting TIC {f.ticid}")
        left_col = col_i == 0
        top_row = row_i == 0
        bottom_row = row_i == nrows-1

        lc_tfa = f.get_lc('tfa', ap=3, clip=3)
        lc_pca = f.get_lc('pca', ap=3, clip=3)
        if ncurves == 1:
            bottom_row = top_row = left_col = True
            plt.close(fig)
            fig = plt.figure(f"Light Curve for TIC {f.ticid}")
            gs = GridSpec(nrows=2, ncols=2, figure=fig)
            lc_ax = fig.add_subplot(gs[0, :])
            pg_ax = fig.add_subplot(gs[1, 0])
            hr_ax = fig.add_subplot(gs[1, 1])
            lc_ax.set_title(f"Light Curves for TFA/PCA Detrending for TIC {f.ticid}")
            lc_ax.set_xlabel("Time (days)")
            lc_ax.set_ylabel("Mag")
            lc_ax.plot(lc_pca.time, lc_pca.flux, label='PCA')
            lc_ax.plot(lc_tfa.time, lc_tfa.flux, label='TFA')
            lc_ax.invert_yaxis()
            lc_ax.legend()

        pg_tfa = lc_tfa.to_periodogram(freq_unit=u.microhertz)
        pg_pca = lc_pca.to_periodogram(freq_unit=u.microhertz)
        # pg_ax.set_title("Power spectrum for TFA/PCA Light Curves")
        pg_ax.set_ylim(0.8e-5, 100e-5)
        pg_ax.set_xlim(2, 100)
        pg_ax.plot(pg_pca.frequency, pg_pca.power, label='PCA')
        pg_ax.loglog(pg_tfa.frequency, pg_tfa.power, label='TFA')
        pg_ax.legend()

        # hr_ax.set_title("Mag-Colour plot of target and dataset")
        hr_ax.plot(bprp, M, 'k.', label='All Stars', markersize=5)
        if index_provided:
            hr_ax.plot(bprp[sub_mask], M[sub_mask], 'b.', label='Indexed region')
        hr_ax.plot(bprp_selection, M_selecion, 'go', label='Selection')
        hr_ax.plot(bprp_selection[i], M_selecion[i], 'ro', label=f'TIC {f.ticid}')
        # For NGC 2446 plots:
        # if True:
        #     hr_ax.set_xlim(0.93, 1.71)
        #     hr_ax.set_ylim(-3.24, 1.11)
            # if bprp_selection[i] > 1.3:
            #     pg_ax.axvline(80, color='k')
            # else:
            #     pg_ax.axvline(25, color='k')
        # mask = table_data[nan_mask]['MatchID'] == int(f.ticid)
        hr_ax.annotate(f"{f.cluster}", xy=(0.99,0.90), ha='right', xycoords='axes fraction')
        hr_ax.annotate(
            f"TMAG {f.TESSmag:.2f}\nTIC {f.ticid}", xy=(0.99, 0.01),
            ha='right', xycoords='axes fraction'
        )


        hr_ax.invert_yaxis()
        # hr_ax.legend()

        if left_col:
            pg_ax.set_ylabel("Power")
            hr_ax.set_ylabel(r"$M_\omega = G + 5 \log_{10}(\omega_{as}) + 5$" + "\n")
        if bottom_row:
            hr_ax.set_xlabel(r"$G_{Bp} - G_{Rp}$")
        if top_row:
            pg_ax.set_xlabel(r"Frequency ($\mu Hz$)")
            if ncurves > 1:
                pg_ax.xaxis.set_label_position('top')


        col_i += 1
        if col_i == ncols:
            col_i = 0
            row_i += 1

    fig.suptitle("Power Spectra and HR Plots")

    mng = plt.get_current_fig_manager()
    mng.window.setVisible(False)
    mng.window.showMaximized()
    plt.show()































#
