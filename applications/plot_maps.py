import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib import colors

divColor = plt.cm.RdBu_r
divColor = plt.cm.seismic
# divColor = plt.cm.PuOr


def main():
    import isceio as io

    folder = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures'

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))

    cbar_kwargs = {
        'orientation': 'vertical',
        'location': 'right',
        'pad': 0.05,
        'aspect': 12,
        'fraction': 0.025,
    }
    tickSize = 8

    # Correlation
    def func(x, pos): return "{:g}".format(np.sign(x) * x**2)
    fmt_correlation = matplotlib.ticker.FuncFormatter(func)
    correlation = np.flip(io.load_file(
        os.path.join(folder, 'correlation.fit')), axis=1)
    correlation_im = ax[0].imshow(
        correlation, norm=colors.CenteredNorm(halfrange=1), cmap=divColor, interpolation='none')
    clb = fig.colorbar(correlation_im, ax=ax[0], format=fmt_correlation, ticks=[
        -1, -0.5, 0, 0.5, 1], **cbar_kwargs)

    clb.ax.tick_params(labelsize=tickSize)
    clb.ax.set_ylabel('R-Squared', fontsize=tickSize)
    clb.ax.yaxis.set_label_position("left")

    ax[0].set_title('(a) R-Squared', loc='left')

    # Slope

    slope = np.flip(io.load_file(
        os.path.join(folder, 'degree_0.fit')), axis=1)

    def func(x, pos): return "{:g}".format(np.sign(x) * x**2)
    fmt_slope = matplotlib.ticker.FuncFormatter(func)
    slope_im = ax[1].imshow(np.sign(
        slope) * np.sqrt(np.abs(slope)), cmap=divColor, norm=colors.CenteredNorm(halfrange=3.5), interpolation='none')

    clb = fig.colorbar(slope_im, ax=ax[1], ticks=[
        -2, -1, 0, 1, 2],
        format=fmt_slope, **cbar_kwargs)
    clb.ax.tick_params(labelsize=tickSize)
    clb.ax.set_ylabel('Slope, m (rad)', fontsize=tickSize)
    # clb.yaxis.set_label_position('left')
    clb.ax.yaxis.set_label_position("left")

    ax[1].set_title('(b) Slope Estimates', loc='left')

    # Intercept

    intercept = np.flip(io.load_file(
        os.path.join(folder, 'degree_1.fit')), axis=1)

    int_im = ax[2].imshow(intercept, norm=colors.CenteredNorm(
        halfrange=0.1), cmap=divColor, interpolation='none')
    clb = fig.colorbar(int_im, ax=ax[2], ticks=[
        -0.1, -0.05, 0, 0.05, 0.1], **cbar_kwargs)

    clb.ax.tick_params(labelsize=tickSize)
    clb.ax.set_ylabel('Intercept, b (rad)', fontsize=tickSize)
    clb.ax.yaxis.set_label_position("left")
    ax[2].set_title('(c) Intercept Estimates', loc='left')

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/dalton_fig.jpg', dpi=300)
    plt.show()


def geo_map():

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from osgeo import gdal
    from numpy import linspace
    from numpy import meshgrid

    ds = gdal.Open(
        "/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures_2l/geo_correlation.fit.vrt")
    data = ds.ReadAsArray()
    geoTransform = ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * ds.RasterXSize
    miny = maxy + geoTransform[5] * ds.RasterYSize
    extent = (minx, miny, maxx, maxy)
    print(extent)
    map = Basemap(projection='tmerc',
                  lat_0=miny, lon_0=minx,
                  llcrnrlon=minx,
                  llcrnrlat=miny,
                  urcrnrlon=maxx,
                  urcrnrlat=maxy)

    # x = linspace(0, map.urcrnrx, data.shape[1])
    # y = linspace(0, map.urcrnry, data.shape[0])

    # xx, yy = meshgrid(x, y)
    def func(x, pos): return "{:g}".format(np.sign(x) * x**2)
    fmt = matplotlib.ticker.FuncFormatter(func)

    map.imshow(np.flip(np.flip(data), axis=1),
               cmap='seismic', vmin=-1, vmax=1)
    map.colorbar(location='bottom',
                 label='R-Squared', format=fmt, ticks=[-1, -0.5, 0, 0.5, 1])
    map.colorbar(location='bottom',
                 label='Correlation', format=fmt, ticks=[-1, -0.5, 0, 0.5, 1])
    map.drawmapscale(-115.37, 36.18, -115.37, 36.4, 5, fontsize=14)

    plt.show()


main()
