import numpy as np
from matplotlib import pyplot as plt
import cartopy
import xarray as xr
import cartopy.crs as ccrs
# import rioxarray
# from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
from matplotlib_scalebar.scalebar import ScaleBar

POIs = {'test2': (68.61790851224154, -149.30411616254935),
        'outcrop_e': (68.620209, -149.293053),
        'polygons_b': (68.623428, -149.312407),
        'polygins_c': (68.622557, -149.317123),
        'outcrop_a': (68.619671, -149.302320),
        'reference': (68.618584, -149.303809),
        'outcrop_c': (68.621911, -149.303943),
        'outcrop_d': (68.62081878634847, - 149.29427851856155),
        'outcrop_f': (68.62055573645934, -149.2942274071651),
        'outcrop_g': (68.62221310244045, -149.31120126598694),
        'outcrop_h': (68.61777651446322, -149.30408567147012),
        'test': (68.62167772076815, -149.30439476233917),
        'outcrop_b': (68.618820, -149.322446),
        'tundra': (68.616262, -149.310665),
        'tundra_b': (68.6285, -149.3048)}

POIs = {
    'Reference': (68.618584, -149.303809),
    'Outcrop A': (68.61777651446322, -149.30408567147012),
    # 'outcrop_b': (68.619037, -149.320111),
    'Outcrop B': (68.621911, -149.303943),
    'Tundra B': (68.62706488016352, -149.30803114685497),
    'Tundra A': (68.622557, -149.317123),
}


def main():
    path = '/Users/rbiessel/Documents/imnav_bing_maps.tif'
    # ds_imnav = xr.open(path, engine='rasterio')

    prj = ccrs.UTM(zone='6')
    srcprj = ccrs.PlateCarree()
    extent = [-149.32395,  -149.278248, 68.615, 68.63]
    lower_utm = prj.transform_point(extent[0], extent[2], srcprj)
    upper_utm = prj.transform_point(extent[1], extent[3], srcprj)
    extent = [lower_utm[0], upper_utm[0], lower_utm[1], upper_utm[1]]

    # fig, ax = plt.subplots(nrows=1, ncols=1, transform=prj)
    ax = plt.axes(projection=prj)

    ds_imnav = xr.open_dataarray(path, engine='rasterio').squeeze()
    ds_imnav = ds_imnav.rio.reproject(prj)
    ds_imnav = ds_imnav/255

    print()

    ax.set_extent(extent, prj)

    ds_imnav.plot.imshow(ax=ax, interpolation=None,
                         transform=prj, extent=extent)

    for key in POIs:
        x, y = prj.transform_point(POIs[key][1], POIs[key][0], srcprj)
        kw = {
            'marker': 'o',
            's': 30,
            'facecolors': 'none',
            'edgecolors': 'white'
        }

        if 'Reference' in key:
            kw['marker'] = '*'
            # kw['edgecolors'] = 'black'

        ax.scatter(x, y, **kw)

        ax.annotate(key, xy=(x, y),
                    textcoords='offset points', xytext=(4, 4), fontsize=12, path_effects=[pe.withStroke(linewidth=1, foreground="white")], alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.add_artist(ScaleBar(
        dx=1,
        units="m",
        dimension="si-length",
        # length_fraction=0.6666,
        fixed_value=100,
        #    scale_formatter=lambda value, unit: f' {value * 1} km ',
        location='lower left',
        box_alpha=0.3
    ))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/imav_test_fig.png',
                dpi=500, transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
