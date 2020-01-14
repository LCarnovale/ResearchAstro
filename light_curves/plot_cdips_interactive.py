import read_cdips as rc
import matplotlib.pyplot as plt
import numpy as np

import time


font = {'weight' : 'normal',
    # 'family' : 'normal',
'size'   : 18}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=False)

# Get all decs and ra's

FITS_COUNT = 500

# all_fits = []
ra_ax = []
dec_ax = []
reprs = []
color_ax = []
mag_ax = []

i = 0
for sec, cams in rc.avail_fits.items():
    for cam, chips in cams.items():
        for chip, fit_list in chips.items():
            for f in fit_list:
                try:
                    new_fit = rc.FitsTable(sec, cam, chip, f)
                except TypeError:
                    print("waiting a bit...")
                    time.sleep(0.5)
                    print("Trying again...")
                    try:
                        new_fit = rc.FitsTable(sec, cam, chip, f)
                    except TypeError:
                        print("Unable to get more fits (got to %d)" % i)
                        i = FITS_COUNT
                        break
                        
                ra_ax.append(new_fit.RA)
                dec_ax.append(new_fit.DEC)
                reprs.append(repr(new_fit))
                if new_fit.BVMAG != 'nan':
                    color_ax.append(float(new_fit.BVMAG))
                    mag_ax.append(new_fit.TESSMAG)
                new_fit.close()
                del new_fit
                i += 1

                if (i >= FITS_COUNT): break
            if (i >= FITS_COUNT): break
        if (i >= FITS_COUNT): break
    if (i >= FITS_COUNT): break

ra_ax = np.array(ra_ax) * np.pi / 180
dec_ax = np.array(dec_ax) * np.pi / 180


plt.figure()
plt.plot(np.array(color_ax), np.array(mag_ax), 'ro')
# plt.show()

plt.figure()
plt.subplot(111, projection='aitoff')
plt.plot(ra_ax, dec_ax, 'r.')

plt.show()

