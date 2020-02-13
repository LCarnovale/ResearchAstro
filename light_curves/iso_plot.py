import matplotlib.pyplot as plt
import numpy as np

"""
Nu max scaling relation:

nu_max = M / ( R**2 * sqrt(T) ) * 3100

All variables are relative to solar values,
solar T(eff) = 5777 K
ie the solar nu_max is 3100 uHz.

"""

iso_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\isoz22sss_eta02_c03hbs\wzsunysunss2.t600700_c03hbs"
iso2_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\isoz32sss_eta02_c03hbs\wz302y288ss2.t600700_c03hbs"
# iso2_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\isoz22s_c03hbs\wzsunysuns.t600700_c03hbs"
ngc_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\ubv.peo"
mast_path = r"C:\Users\Leo\OneDrive - UNSW\Uni\ResearchStello\light_curves\MAST_Crossmatch_NGC_5822.csv"

dist_mod = 10.22
age = 10**8.588 # 10**8.588 = 0.387 Gyr

font = {'weight' : 'normal',
# 'family' : 'normal',
'size'   : 17}
plt.rc('font', **font)
plt.rc('figure', figsize=(12, 8))
plt.rc('axes', grid=True)
plt.rc('lines', markersize=10)

iso_names = [
    "M_in", "M", "log_L", "log_T", "Mv", "UB", "BV", "VI", "VR", "VJ", "VK", "VL", "HK"
]



iso = np.genfromtxt(iso_path, names=iso_names)
iso2 = np.genfromtxt(iso2_path, names=iso_names)
ngc = np.genfromtxt(ngc_path, names=True, delimiter='\t')
mast = np.genfromtxt(mast_path, names=True, delimiter=',', skip_header=4)


plt.subplot(121)
plt.plot(iso['BV'] + iso['VR'], iso['Mv'], '.-', label=r'Iso model $t=0.7$ Gyr')
# plt.plot(iso2['BV'] + iso2['VR'], iso2['Mv'], 'b.-', label=r'Iso model $\eta=0.4$')
gaia_M = mast['phot_g_mean_mag'] + 5*np.log10(mast['parallax']/1e3) + 5
plt.plot(mast['bp_rp'], gaia_M, 'go', label='Gaia NGC 2447')
plt.plot((iso['BV'] + iso['VR'])[::100], iso['Mv'][::100], 'r^', label="Iso model, every 100th step")
plt.plot((iso['BV'] + iso['VR'])[::500], iso['Mv'][::500], 'k^')
plt.xlabel("BR")
plt.ylabel("M")
# plt.plot(ngc['BV'], ngc['V'], 'bo', label='WEBDA NGC 2447')
plt.legend()
plt.gca().invert_yaxis()

L_ax = 10**iso['log_L']
M_ax = iso['M']
T_ax = 10**iso['log_T']
R_ax = np.sqrt(L_ax / (T_ax/5777)**4)
nu_max_ratio = M_ax * (T_ax / 5777)**3.5 / L_ax

x_ax = np.linspace(0, 1, len(iso))
plt.subplot(322)
plt.title("Mass, L, R along isochrone")
plt.ylabel(r"$X / X_{\odot}$")
plt.semilogy(x_ax, M_ax, label='M')
plt.plot(x_ax[::100], M_ax[::100], 'r^')
plt.plot(x_ax[::500], M_ax[::500], 'k^')

plt.plot(x_ax, L_ax, label='L')
plt.plot(x_ax[::100], L_ax[::100], 'r^')
plt.plot(x_ax[::500], L_ax[::500], 'k^')

plt.plot(x_ax, R_ax, label='R')
plt.plot(x_ax[::100], R_ax[::100], 'r^')
plt.plot(x_ax[::500], R_ax[::500], 'k^')

plt.legend()

plt.subplot(324)
plt.title("Teff along isochrone")
plt.ylabel("T [K]")
plt.plot(x_ax, T_ax)
plt.plot(x_ax[::100], T_ax[::100], 'r^')
plt.plot(x_ax[::500], T_ax[::500], 'k^')
plt.subplot(326)
plt.title(r"$\nu_{max}$ along isochrone")
# plt.ylabel(r"$\nu_{max} / \nu_{max, \odot}$")
plt.ylabel(r"$\nu_{max}$ ($\mu Hz$), [$\nu_{max, \odot} = 3100 \mu Hz$]")
nu_max = nu_max_ratio * 3100
plt.semilogy(x_ax, nu_max)
plt.plot(x_ax[::100], nu_max[::100], 'r^')
plt.plot(x_ax[::500], nu_max[::500], 'k^')
plt.gca().invert_yaxis()


    # plt.plot(1, 1)


plt.show()
