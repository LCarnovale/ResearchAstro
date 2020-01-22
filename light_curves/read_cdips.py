from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import pickle
import sys
from astroquery.mast import Catalogs

# index_file = "cdips_index.pickle"

# data_root = r"Y:\Uni\ResearchStello"
data_root = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello"

data_folder = data_root + r"\CDIPS"
index_file = data_root + r"\cdips_index.pickle"

time_name = "TMID_BJD" # Exp mid-time in BJD_TDB (BJDCORR applied)

sample_id = "0003016249811363341568"

t_step_mins = 30 # Minutes between samples
MINS_IN_DAY = 1440
SECS_IN_DAY = 60 * MINS_IN_DAY

MAX_INDEX = 5000

def sstr(num):
    return "hlsp_cdips_tess_ffi_gaiatwo%022d"%num

def build_index(max_files=MAX_INDEX):
    avail_fits = {} # {sector:{cam:{chip:[ids], ... all chips}, ... all cams}, ... all sectors}
    # Get available ids
    from os import listdir
    # Check sectors:
    sectors = listdir(data_folder)
    n = 0
    print("Indexing CDIPS file...")
    for s in sectors:
        sec = int(s[1:])
        avail_fits[sec] = {}
        sec_path = data_folder + "\\" + s
        camchips = listdir(sec_path)
        for camchip in camchips:
            cam, chip = camchip.split('_')
            cam = int(cam[3:])
            chip = int(chip[3:])
            if cam not in avail_fits[sec]:
                avail_fits[sec][cam] = {}
            d = avail_fits[sec][cam]
            d[chip] = []
            camchip_path = sec_path + "\\" + camchip
            fits_files = listdir(camchip_path)
            for f in fits_files:
                temp = f.split('gaiatwo')[1]
                gid = temp.split('-')[0]
                d[chip].append(gid)
                n += 1
                print(
                    "Fits files found: %d / %d" % (n, max_files),
                    end="\r", flush=True
                )

                if n > max_files: break
            if n > max_files: break
        if n > max_files: break

    print()
    print("pickling index...")

    with open(index_file, 'wb') as f:
        pickle.dump(avail_fits, f)

    print("done.")


def load_index():
    print("Loading index...")
    with open(index_file, 'rb') as f:
        avail_fits = pickle.load(f)

    print("Done.")
    return avail_fits

if 'index' in sys.argv:
    build_index()

avail_fits = load_index()

def get_obs_id(sector, cam, ccd, gaiaid):
    """ This can be used with:
        from astroquery.mast import Observations
        Observations.query_criteria(obs_id=<output from this>)

    To get more information about the observation.
    """
    sector_str = f"{sector:0>4}"
    obs_id = f"hlsp_cdips_tess_ffi_gaiatwo{gaiaid}-{sector_str}-cam{cam}-ccd{ccd}_tess_v01_llc"
    return obs_id

def get_fits(sector, cam, ccd, gaiaid):
    """ Return an opened fits file with the given source properties.
    `gaiaid` can either be the actual id (a string) or an integer
    index which will be given to the avail_fits index to lookup the desired
    id string.
    """

    if type(gaiaid) == int:
        try:
            gaiaid = avail_fits[sector][cam][ccd][gaiaid]
        except KeyError:
            raise KeyError(f"index[{sector}][{cam}][{ccd}][{gaiaid}] does not exist.")


    sector_str = f"{sector:0>4}"
    cam_str = f"cam{cam}_ccd{ccd}"
    filename = get_obs_id(sector, cam, ccd, gaiaid) + ".fits"

    path = rf"{data_folder}\s{sector_str}\{cam_str}\{filename}"

    try:
        hdul = fits.open(path)
    except FileNotFoundError:
        print("fits file not found:")
        print(path)
        print("Arguments:")
        print(f"sector: {sector}, cam: {cam}, ccd: {ccd}, gaiaid: {gaiaid}")
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
    def __init__(self, sector, cam, ccd, gaiaid):
        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.gaiaid = gaiaid
        self._hdul = get_fits(sector, cam, ccd, gaiaid)
        self._data = self._hdul[1].data
        self._open = True

        self._pca_lcs = {}
        self._tfa_lcs = {}


    def __str__(self):
        return f"""sector  : {self.sector}
camera  : {self.cam}
chip/ccd: {self.ccd}
gaiaid  : {self.gaiaid}"""

    def __repr__(self):
        return f"FitsTable({self.sector}, {self.cam}, {self.ccd}, '{self.gaiaid}')"

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

    # def get_tfa_mag(self, ap=1):
    #     # ap (aperture) can be 1, 2 or 3
    #     return self._data[f'TFA{ap}'].copy()

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
        `ap` should be 1 (default) 2 or 3.
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
            t_ax = self.get_time_ax()
            if clip:
                mask = sigma_clip(flux, clip)
                flux = flux[mask]
                t_ax = t_ax[mask]
            lc = lk.LightCurve(time=t_ax, flux=flux)
            lcs[ap] = lc
        return lc

    # def get_tfa_lc(self, type, ap=1, clip=None):

    #     if ap in self._tfa_lcs:
    #         lc = self._tfa_lcs[ap]
    #     else:
    #         lc = lk.LightCurve(time=self.get_time_ax(), flux=self.get_tfa_mag(ap))
    #         self._tfa_lcs[ap] = lc
    #     return lc

    def get_ft(self, source_type, ap=1, clip=None, force_recalc=False):
        lc = self.get_lc(source_type, ap, clip, force_recalc)
        lc.flux *= -1
        pg = lc.to_periodogram()
        freq = pg.frequency
        power = pg.power
        return (freq, power)

    def query_mast(self):
        """ Fetch the data for this star in astroquery.mast using the TICID
        """
        t = Catalogs.query_criteria(target_name=self.ticid)
        return t


    @property
    def cdipsref(self):
        return self._hdul[0].header['CDIPSREF']

    @property
    def ticid(self):
        return self._hdul[0].header['TICID']

    @property
    def RA(self):
        return self._hdul[0].header['BTC_RA'] # Right ascen in barycentric time correction

    @property
    def DEC(self):
        return self._hdul[0].header['BTC_DEC'] # Declination in barycentric time correction

    @property
    def TESSMAG(self):
        return self._hdul[0].header['TESSMAG']

    @property
    def GMAG(self):
        return self._hdul[0].header['phot_g_mean_mag'] # assuming this is the colour index we're looking for

    @property
    def BMAG(self):
        return self._hdul[0].header['phot_bp_mean_mag'] # assuming this is the colour index we're looking for

    @property
    def RMAG(self):
        return self._hdul[0].header['phot_rp_mean_mag'] # assuming this is the colour index we're looking for


if __name__ == "__main__":

    font = {'weight' : 'normal',
    # 'family' : 'normal',
    'size'   : 18}
    plt.rc('font', **font)
    plt.rc('figure', figsize=(12, 8))
    plt.rc('axes', grid=True)

    print("Plotting sample light curve with FFT")


    for target in avail_fits[6][1][1][10:15]:
        ft = FitsTable(6, 1, 1, target)
        t_ax = ft.get_time_ax() # in days

        fig_title = f"TFA vs PCA light curves and fourier transforms for {target}."
        print(fig_title)
        fig = plt.figure(fig_title)
        aperture_sizes = [3]
        PCA_N = [ft.get_lc('pca', n, 3) for n in aperture_sizes]
        TFA_N = [ft.get_lc('tfa', n, 3) for n in aperture_sizes]

        # (pca_lc, tfa_lc), (pca_ft, tfa_ft) = fig.subplots(nrows=2, ncols=2, sharey='row', sharex='row')
        lc_ax, ft_ax = fig.subplots(nrows=2)
        tfa_lc = pca_lc = lc_ax
        tfa_ft = pca_ft = ft_ax

        # Sharing y-axis hides tick labels by default.
        # This puts them back:
        tfa_lc.yaxis.set_tick_params(labelleft=True)
        tfa_ft.yaxis.set_tick_params(labelleft=True)

        for lc in PCA_N:
            t_ax = lc.time
            flux = lc.flux
            pca_lc.plot(t_ax, flux)

        for lc in TFA_N:
            t_ax = lc.time
            flux = lc.flux
            tfa_lc.plot(t_ax, flux)


        # aperture_legend = ['1 px', '1.5 px', '2.25 px'][2:]

        pca_lc.set_title("PCA/TFA detrending light curves")

        # tfa_lc.set_title("TFA detrending light curves")

        for ax in (pca_lc, tfa_lc):
            ax.set_ylabel("Mag")
            ax.set_xlabel("time (days)")
            ax.legend(['PCA', 'TFA'], title='Detrending methods')

        # Fourier transforms:
        size = len(PCA_N[0])

        pca_freq_power = np.array([np.array(ft.get_ft('pca', n, 3)) for n in (1, 2, 3)]) # shape (3, 2, `size`)
        # pca_freq_power = np.log10(pca_freq_power)
        pca_freq = pca_freq_power[0, 0, :]
        pca_power = pca_freq_power[2, 1, :]

        tfa_freq_power = np.array([np.array(ft.get_ft('tfa', n, 3)) for n in (1, 2, 3)]) # shape (3, 2, `size`)
        # tfa_freq_power = np.log10(tfa_freq_power)
        tfa_freq = tfa_freq_power[0, 0, :]
        tfa_power = tfa_freq_power[2, 1, :]

        pca_ft.loglog(pca_freq, pca_power.T)
        pca_ft.set_title("PCA Fourier Transforms")
        tfa_ft.loglog(tfa_freq, tfa_power.T)
        tfa_ft.set_title("TFA Fourier Transforms")

        for ax in (pca_ft, tfa_ft):
            ax.set_xlabel(r"$\log_{10}(\nu [\mu Hz] )$")
            ax.set_ylabel(r"$\log_{10}$ Power")
            pk = np.max(pca_power)
            mn = np.min(pca_power)
            # ax.set_ylim(mn - .1*abs(mn), pk + .1*abs(pk))

    plt.show()
