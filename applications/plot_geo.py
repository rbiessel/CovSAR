import matplotlib
import xarray as xr
import rioxarray
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import urllib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
from matplotlib.cm import get_cmap
import os
from scipy.ndimage.morphology import binary_erosion as erode
from scipy import ndimage, misc
import matplotlib.ticker as ticker

globfigparams = {
    'fontsize': 10, 'family': 'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
    'column_inch': 229.8775 / 72.27, 'markersize': 24, 'markercolour': '#AA00AA',
    'fontcolour': 'black', 'tickdirection': 'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1}

colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']
plt.rc('font', **
       {'size': globfigparams['fontsize']})
plt.rcParams['text.usetex'] = globfigparams['usetex']
plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
plt.rcParams['font.size'] = globfigparams['fontsize']
plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
plt.rcParams['xtick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']


stack = 'vegas'

# Load Data
if 'vegas' in stack:
    base_path = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures_dif_MLE'

if 'dalton' in stack:
    base_path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_dif_MLE'


int_ds = xr.open_dataarray(os.path.join(
    base_path, 'geo_degree_1.fit.vrt'), engine='rasterio').squeeze()

r_ds = xr.open_dataarray(os.path.join(
    base_path, 'geo_correlation.fit.vrt'), engine='rasterio').squeeze()

slope_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_degree_0.fit.vrt'), engine='rasterio').squeeze()

dif_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_cumulative_difference.fit.vrt'), engine='rasterio').squeeze()

coh_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_average_coherence.fit.vrt'), engine='rasterio').squeeze()


dsets = [int_ds, r_ds, slope_ds, dif_ds, coh_ds]
utm = True

if utm:
    proj = int_ds.rio.estimate_utm_crs()
    int_ds = int_ds.rio.reproject(proj)
    r_ds = r_ds.rio.reproject(proj)
    slope_ds = slope_ds.rio.reproject(proj)
    dif_ds = dif_ds.rio.reproject(proj)
    coh_ds = coh_ds.rio.reproject(proj)

# Kwargs

cbar_kwargs = {
    'orientation': 'vertical',
    'pad': 0.05,
    'aspect': 12,
    'fraction': 0.025,
    'shrink': 0.5,
}

cmap_div = get_cmap('cet_diverging_bwr_20_95_c54')
cmap_cont = get_cmap('cet_linear_grey_0_100_c0')
cmap_cont = cc.cm.fire
cmap_cont = get_cmap('cet_linear_protanopic_deuteranopic_kbw_5_98_c40')
cmap_cont = get_cmap('viridis')

latmin = float(int_ds.y.min())
latmax = float(int_ds.y.max())

lonmin = float(int_ds.x.min())
lonmax = float(int_ds.x.max())

x = int_ds.x.to_numpy()
y = int_ds.y.to_numpy()


if 'vegas' in stack:
    landsat_ds = xr.open_dataarray('/Users/rbiessel/Documents/visibleData/falseColor.tif',
                                   engine='rasterio', mask_and_scale=False)

if 'dalton' in stack:
    landsat_ds = xr.open_dataarray('/Users/rbiessel/Documents/visibleData/dalton_falseColor.tif',
                                   engine='rasterio', mask_and_scale=False)

dsets.append(landsat_ds)

if utm:
    landsat_ds = landsat_ds.rio.reproject(proj)


# proj = ccrs.PlateCarree()
ls_subset = landsat_ds.rio.clip_box(minx=lonmin, miny=latmin,
                                    maxx=lonmax, maxy=latmax)

landsat = ls_subset.to_numpy()
landsat = landsat/landsat.max()

landsat = np.flip(landsat, 0)
landsat = np.transpose(landsat, [1, 2, 0]) * 3


intercept = int_ds.to_numpy()
slope = slope_ds.to_numpy()
rsquared = r_ds.to_numpy()
difdisp = dif_ds.to_numpy()
meanCoherence = coh_ds.to_numpy()

buffer = 2000

extent = [lonmin, lonmax, latmin, latmax]

fig, ax = plt.subplots(
    ncols=2, nrows=2, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)

print(landsat.shape)


alpha = np.ones(intercept.shape)
mask = np.where(intercept == 0)
mask2 = np.where(intercept > 100)

alpha[mask] = 0
alpha[mask2] = 0

print(alpha)
alpha = erode(alpha, np.ones((30, 30))).astype(np.float32)


def plot_optical(ax, data, alpha=0.9):
    return ax.imshow(data, extent=extent, interpolation='none', alpha=alpha)


# Landsat/Sentinel-2 Image
thisax = ax[0, 0]
lsim = plot_optical(thisax, landsat, alpha=alpha)
divider = make_axes_locatable(thisax)
caxls = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(lsim, cax=caxls)
thisax.set_title('(a) Sentinel-2 False-Color, Bands 8, 4, \& 3', loc='left')

# R-Squared

thisax = ax[0, 1]
lsim = plot_optical(thisax, landsat, alpha=0.9)
norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=1)


rim = thisax.imshow(rsquared**2, extent=extent,
                    cmap=cmap_cont, norm=norm, alpha=alpha, interpolation='none')

divider = make_axes_locatable(thisax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(rim, cax=cax, **cbar_kwargs)
cbar.ax.set_title('[$-$]')

thisax.set_title('(b) R-Squared', loc='left')

# Slope
thisax = ax[1, 0]
lsim = plot_optical(thisax, landsat, alpha=0.9)
norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
slopem = thisax.imshow(np.sign(slope) * np.sqrt(np.abs(slope)), extent=extent,
                       cmap=cmap_div, norm=norm, interpolation='none', alpha=alpha)
divider = make_axes_locatable(thisax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(slopem, cax=cax)
cbar.ax.set_title('[$rad$]')

thisax.set_title('(c) Slope', loc='left')

# Intercept

thisax = ax[1, 1]
lsim = plot_optical(thisax, landsat, alpha=0.9)


norm = matplotlib.colors.Normalize(vmin=-.1, vmax=.1)
intim = thisax.imshow(intercept, extent=extent,
                      cmap=cmap_div, alpha=alpha, norm=norm, interpolation='none')

divider = make_axes_locatable(thisax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(intim, cax=cax)
cbar.ax.set_title('[$rad$]')

thisax.set_title('(d) Mean Residual Closure Phase', loc='left')


# Difference
if False:
    thisax = ax[1, 1]
    # lsim = plot_optical(thisax, landsat)
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
    difIm = thisax.imshow(difdisp, extent=extent,
                          cmap=cmap_div, alpha=alpha, norm=norm, interpolation='none')
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = fig.colorbar(difIm, cax=cax)
    cbar.ax.set_title('[$mm$]')
    thisax.set_title('(e) End-of-Stack Displacement Difference', loc='left')


# Mean Coherence
if False:
    thisax = ax[1, 2]
    # lsim = plot_optical(thisax, landsat)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cohIm = thisax.imshow(meanCoherence, extent=extent,
                          cmap=cmap_cont, alpha=alpha, norm=norm, interpolation='none')
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = fig.colorbar(cohIm, cax=cax)
    cbar.ax.set_title('[$-$]')
    thisax.set_title('(e) Average Nearest-Neighbor \n Coherence', loc='left')


# Get rid of landsat colorbar
caxls.remove()

# Grids
for axis in ax.flatten():
    axis.set_xlabel(r'Eastings [$km$]')
    axis.set_ylabel(r'Northings [$km$]')
    axis.grid(alpha=0.5)
    width = intercept.shape[1]
    height = intercept.shape[0]
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
    axis.xaxis.set_major_formatter(ticks)
    axis.yaxis.set_major_formatter(ticks)

    axis.set_xlim((lonmin + buffer, lonmax - buffer))
    axis.set_ylim((latmin + buffer, latmax - buffer))

plt.tight_layout()
plt.savefig(f'/Users/rbiessel/Documents/maps_{stack}.jpg', dpi=300)
plt.show()
