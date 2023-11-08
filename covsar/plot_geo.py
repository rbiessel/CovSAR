from pub_pixels import pixel_paths
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
from scipy.ndimage import binary_erosion as erode
from scipy import ndimage, misc
import matplotlib.ticker as ticker
import figStyle
import matplotlib.patheffects as pe


import glob

paths = glob.glob('/Users/rbiessel/Documents/*.tif')

# for file_path in paths:
#     # load path

stack = 'vegas'

# Load Data
if 'vegas' in stack:
    base_path = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures'
    intercept_range = 0.07
    slope_range = 0.7
    disp_dif_range = 1
    bias_range = 0.07
    optical_label = '(a) Sentinel-2 False-Color:\n Bands 8, 4, \& 3'

    labels = {
        'CC': {'pos': [4034, 644], 'offset': [-25, -30]},
        'AC': {'pos': [4029, 653], 'offset': [15, 15]},
        'LV': {'pos':  [4009, 661], 'offset': [0, 0]},
    }

    pixel_paths = pixel_paths[0:3]
    pixel_labels = ['(a)', '(b)', '(c)']

'/Users/rbiessel/Documents/InSAR/plotData/DNWR/p_116_54/',
'/Users/rbiessel/Documents/InSAR/plotData/vegas_east/p_34_17/'

if 'dalton' in stack:
    base_path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures'
    intercept_range = 1
    slope_range = 1.75
    disp_dif_range = 3
    bias_range = 0.7
    optical_label = '(a) Sentinel-2 True-Color'

    labels = {
        'BR': {'pos': [7593, 406], 'offset': [0, 0]},
        'SM': {'pos': [7629, 412], 'offset': [0, 0]},
        'DH': {'pos': [7618, 408], 'offset': [0, 0]},
    }

    pixel_paths = pixel_paths[3:]
    pixel_labels = ['(d)', '(e)', '(f)']

# '/Users/rbiessel/Documents/InSAR/plotData/dalton/p_171_366/'

int_ds = xr.open_dataarray(os.path.join(
    base_path, 'geo_degree_1.fit.vrt'), engine='rasterio').squeeze()

r_ds = xr.open_dataarray(os.path.join(
    base_path, 'geo_correlation.fit.vrt'), engine='rasterio').squeeze()

slope_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_degree_0.fit.vrt'), engine='rasterio').squeeze()

max_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_max_difference.fit.vrt'), engine='rasterio').squeeze()

bias_ds = xr.open_dataarray(
    os.path.join(base_path, 'geo_bias.fit.vrt'), engine='rasterio').squeeze()

# coh_ds = xr.open_dataarray(
#     os.path.join(base_path, 'geo_average_coherence.fit.vrt'), engine='rasterio').squeeze()


dsets = [int_ds, r_ds, slope_ds, max_ds, bias_ds]

proj = int_ds.rio.estimate_utm_crs()
print(proj)
int_ds = int_ds.rio.reproject(proj, 100)
r_ds = r_ds.rio.reproject(proj, 100)
slope_ds = slope_ds.rio.reproject(proj, 100)
max_ds = max_ds.rio.reproject(proj, 100)
bias_ds = bias_ds.rio.reproject(proj, 100)

# Kwargs

cbar_kwargs = {
    'orientation': 'vertical',
    'pad': 0.1,
    'aspect': 12,
    'fraction': 0.025,
    'shrink': 0.5,
}

title_pad = 10

cmap_div = get_cmap('cet_diverging_bwr_20_95_c54')
cmap_cont = get_cmap('cet_linear_grey_0_100_c0')
cmap_cont = cc.cm.fire
cmap_cont = get_cmap('cet_linear_protanopic_deuteranopic_kbw_5_98_c40')
cmap_cont = get_cmap('viridis')

latmin = float(int_ds.y.min())
latmax = float(int_ds.y.max())

lonmin = float(int_ds.x.min())
lonmax = float(int_ds.x.max())

# x = int_ds.x.to_numpy()
# y = int_ds.y.to_numpy()


if 'vegas' in stack:
    landsat_ds = xr.open_dataarray('/Users/rbiessel/Documents/visibleData/falseColor.tif',
                                   engine='rasterio', mask_and_scale=False)

if 'dalton' in stack:
    landsat_ds = xr.open_dataarray('/Users/rbiessel/Documents/visibleData/dalton_trueColor.tif',
                                   engine='rasterio', mask_and_scale=False)


dsets.append(landsat_ds)
landsat_ds = landsat_ds.rio.reproject(proj)


# proj = ccrs.PlateCarree()
ls_subset = landsat_ds.rio.clip_box(minx=lonmin, miny=latmin,
                                    maxx=lonmax, maxy=latmax)

landsat = ls_subset.to_numpy()

if 'vegas' in stack:
    landsat = landsat/landsat.max()
    landsat = np.flip(landsat, 0)
    landsat = np.transpose(landsat, [1, 2, 0])
    for band in range(3):
        landsat[:, :, band] = landsat[:, :, band]**1.3
    # Scale RED
    landsat[:, :, 0] *= 5
    # Scale GREEN
    landsat[:, :, 1] *= 5
    # Scale BLUE
    landsat[:, :, 2] *= 5

if 'dalton' in stack:
    landsat = landsat/0.04
    # landsat = landsat * 5
    # landsat = np.flip(landsat, 0)
    landsat = np.transpose(landsat, [1, 2, 0])

    for band in range(3):
        landsat[:, :, band] = landsat[:, :, band]**1.2
    # Scale RED
    landsat[:, :, 0] *= 2.5
    # Scale GREEN
    landsat[:, :, 1] *= 2.5
    # Scale BLUE
    landsat[:, :, 2] *= 2.5


# Convert Rest of data to Numpy Arrays
intercept = int_ds.to_numpy()
slope = slope_ds.to_numpy()
rsquared = r_ds.to_numpy()
max_dif = max_ds.to_numpy()
bias = bias_ds.to_numpy()

buffer = 100

extent = [lonmin, lonmax, latmin, latmax]
print(extent)

if 'vegas' in stack:
    fig, ax = plt.subplots(
        ncols=2, nrows=3, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True, transform=proj)

    prj = ccrs.UTM(zone='11')

elif 'dalton' in stack:
    fig, ax = plt.subplots(
        ncols=3, nrows=2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True, transform=proj)
    prj = ccrs.UTM(zone='6')


alpha = np.ones(intercept.shape)
mask = np.where(intercept == 0)
mask2 = np.where(intercept > 100)

alpha[mask] = 0
alpha[mask2] = 0

print(alpha)
alpha = erode(alpha, np.ones((10, 10))).astype(np.float32)
alpha = alpha


def plot_optical(ax, data, alpha=0.9):
    return ax.imshow(data, extent=extent, interpolation='none', alpha=alpha, clim=(0, 175))


stroke_effect = [pe.withStroke(linewidth=5, foreground="white", alpha=0.95)]
stroke_effect_line = [pe.withStroke(
    linewidth=3, foreground="white", alpha=0.6)]

# Landsat/Sentinel-2 Image
thisax = ax[0, 0]
lsim = plot_optical(thisax, landsat, alpha=alpha)
# label image with locations
# for each item in dictionary labels
for key in labels:
    loc = labels[key]['pos']
    offset = labels[key]['offset']
    # thisax.text(loc[1], loc[0], key,
    #             transform=thisax.transform, path_effects=stroke_effect)

    thisax.annotate(key, xy=(loc[1] * 1e3, loc[0] * 1e3),
                    textcoords='offset points', xytext=offset, fontsize=14,
                    path_effects=stroke_effect, alpha=1, color='darkslategray', arrowprops=dict(facecolor='white', arrowstyle='-', color='darkslategray', path_effects=stroke_effect_line))

for i in range(len(pixel_paths)):
    # load latlon from np file
    latlon = np.load(os.path.join(pixel_paths[i], 'latlon.np.npy'))
    plabel = pixel_labels[i]
    srcprj = ccrs.CRS('EPSG:4326')
    loc = prj.transform_point(latlon[1], latlon[0], srcprj)
    thisax.scatter(loc[0], loc[1], marker='*', c='white', s=30, alpha=0.95)
    thisax.annotate(plabel, xy=(loc[0], loc[1]),
                    textcoords='offset points', xytext=(-21, 6), fontsize=12, path_effects=stroke_effect, alpha=1, color='darkslategray')

divider = make_axes_locatable(thisax)
caxls = divider.append_axes('right', size='5%', pad=0.08)
fig.colorbar(lsim, cax=caxls)
thisax.set_title(optical_label, loc='left')

# R-Squared


thisax = ax[0, 1]
if 'vegas' in stack:
    thisax = ax[0, 1]
# lsim = plot_optical(thisax, landsat, alpha=0.9)
norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=1)

rim = thisax.imshow(rsquared**2, extent=extent,
                    cmap=cmap_cont, norm=norm, alpha=alpha, interpolation='none')

divider = make_axes_locatable(thisax)
cax = divider.append_axes('right', size='5%', pad=0.08)
cbar = fig.colorbar(rim, cax=cax, **cbar_kwargs)
cbar.ax.set_title('[$-$]', pad=title_pad)

thisax.set_title('(b) $R^2$', loc='left')

# Slope
if 'vegas' in stack:
    thisax = ax[1, 0]
else:
    thisax = ax[0, 2]


# lsim = plot_optical(thisax, landsat, alpha=0.9)
norm = matplotlib.colors.Normalize(vmin=-slope_range, vmax=slope_range)
slopem = thisax.imshow(np.sign(slope) * np.sqrt(np.abs(slope)), extent=extent,
                       cmap=cmap_div, norm=norm, interpolation='none', alpha=alpha)
divider = make_axes_locatable(thisax)


def func(x, pos): return "{:g}".format(np.sign(x) * x)


fmt_slope = matplotlib.ticker.FuncFormatter(func)

cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(slopem, cax=cax, ticks=[-slope_range, -slope_range/2, 0, slope_range/2, slope_range],
                    format=fmt_slope,)
cbar.ax.set_title('[$\mathrm{rad}$]', pad=title_pad)

thisax.set_title('(c) $m$', loc='left')

# Intercept

if 'vegas' in stack:
    thisax = ax[1, 1]
else:
    thisax = ax[1, 0]

# lsim = plot_optical(thisax, landsat, alpha=0.9)

norm = matplotlib.colors.Normalize(vmin=-intercept_range, vmax=intercept_range)
intim = thisax.imshow(intercept, extent=extent,
                      cmap=cmap_div, alpha=alpha, norm=norm, interpolation='none')

divider = make_axes_locatable(thisax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(intim, cax=cax, ticks=[-intercept_range, -
                                           intercept_range/2, 0, intercept_range/2, intercept_range])
cbar.ax.set_title('[$\mathrm{rad}$]', pad=title_pad)

thisax.set_title(
    '(d) $b$', loc='left')

# Difference
if True:
    if 'vegas' in stack:
        thisax = ax[2, 0]
    else:
        thisax = ax[1, 1]

    # lsim = plot_optical(thisax, landsat)
    norm = matplotlib.colors.Normalize(
        vmin=-disp_dif_range, vmax=disp_dif_range)
    difIm = thisax.imshow(max_dif, extent=extent,
                          cmap=cmap_div, alpha=alpha, norm=norm, interpolation='none')
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(
        difIm, cax=cax, ticks=[-disp_dif_range, -disp_dif_range/2, 0, disp_dif_range/2, disp_dif_range])
    cbar.ax.set_title('[$\mathrm{mm}$]', pad=title_pad)
    thisax.set_title(
        '(e) '
        r'$\Delta \theta_{\mathrm{smax}}$', loc='left')


# Rate Bias
if True:
    if 'vegas' in stack:
        thisax = ax[2, 1]
    else:
        thisax = ax[1, 2]

    # lsim = plot_optical(thisax, landsat)
    norm = matplotlib.colors.Normalize(
        vmin=-bias_range, vmax=bias_range)
    biasIm = thisax.imshow(bias * 365 / 12, extent=extent,
                           cmap=cmap_div, alpha=alpha, norm=norm, interpolation='none')
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(
        biasIm, cax=cax, ticks=[-bias_range, -bias_range/2, 0, bias_range/2, bias_range])
    cbar.ax.set_title('[$\mathrm{mm}/\mathrm{month}$]', pad=title_pad)
    thisax.set_title(r'(f) $\Delta \dot{ \theta}$', loc='left')


# Get rid of landsat colorbar
caxls.remove()

# Grids
for axis in ax.flatten():
    axis.set_xlabel(r'Eastings [$\mathrm{km}$]')
    axis.set_ylabel(r'Northings [$\mathrm{km}$]')
    axis.grid(alpha=0.3)
    width = intercept.shape[1]
    height = intercept.shape[0]
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
    axis.xaxis.set_major_formatter(ticks)
    axis.yaxis.set_major_formatter(ticks)

    axis.set_xlim((lonmin + buffer, lonmax - buffer))
    axis.set_ylim((latmin + buffer, latmax - buffer))

plt.tight_layout()
plt.savefig(
    f'/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/maps_{stack}.png', dpi=500)
plt.show()
