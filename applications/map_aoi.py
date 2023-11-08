# EOmaps example 3: Customize the appearance of the plot

# from eomaps import Maps
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt

# crs = Maps.CRS.Orthographic(-127.305625, 47.207366)
# m = Maps(crs=crs, figsize=(9, 5))

# # m.text(0.5, 0.97, "What a nice figure", fontsize=12)


# imnav = (68.642262, -149.356780)
# vegas = (36.316504, -115.262798)


# # m.add_feature.preset.coastline()
# m.add_feature.preset.ocean(facecolor='#457b9d')
# m.add_feature.preset.land(facecolor='#edede9', edgecolor='black')
# # m.add_feature.preset.states(alpha=0.1)

# # m.add_feature.preset.rivers_lake_centerlines()

# m.add_feature.preset.countries(alpha=0.5)
# # m.add_feature.preset.urban_areas()

# # m.set_data([1, 0], x=[imnav[1], vegas[1]], y=[imnav[0], vegas[0]],
# #            crs=4326)

# m.add_marker(xy=(imnav[1], imnav[0]), xy_crs=4326, buffer=10, radius='pixel',
#              fc="black", lw=1, alpha=1)


# # m.add_marker(xy=(imnav[1], imnav[0]), xy_crs=4326, radius='pixel', shape="ellipses",
# #              fc="black", lw=1, alpha=1)

# # m.set_data(data=data, x="lon", y="lat", crs=4326)
# # plot geodesic-circles with 30 km radius
# # m.set_shape.geod_circles(radius=30000)
# # m.set_classify_specs(
# #     scheme="UserDefined", bins=[35, 36, 37, 38, 45, 46, 47, 48, 55, 56, 57, 58]
# # )

# # m.plot_map()
# # m.plot_map(
# #     edgecolor="k",  # give shapes a black edgecolor
# #     linewidth=0.5,  # with a linewidth of 0.5
# #     cmap="RdYlBu",  # use a red-yellow-blue colormap
# #     vmin=35,  # map colors to values between 35 and 60
# #     vmax=60,
# #     alpha=0.5,  # add some transparency
# #     # facecolor='white'
# # )
# # mp.set_patch_props(fc="none", ec="none", offsets=(1, 1.6, 1, 1))


# # add a colorbar
# # m.add_colorbar(
# #     label="some parameter",
# #     hist_bins="bins",
# #     hist_size=1,
# #     hist_kwargs=dict(density=True),
# # )

# # add a y-label to the histogram
# # m.colorbar.ax_cb_plot.set_ylabel("The Y label")

# # m.apply_layout(
# #     {
# #         "figsize": [9.0, 5.0],
# #         "0_map": [0.10154, 0.2475, 0.79692, 0.6975],
# #         "1_cb": [0.20125, 0.0675, 0.6625, 0.135],
# #         "1_cb_histogram_size": 1,
# #         "2_logo": [0.87501, 0.09, 0.09999, 0.07425],
# #     }
# # )
# # m.f.set_facecolor('white')
# # m.ax.set_facecolor('white')
# # m.f.patch.set_alpha(1)
# # m.draw.polygon()

# m.show()

import matplotlib.pyplot as plt
import numpy as np

import cartopy
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def main():

    AOIs = {
        "Dalton Highway": (68.642262, -149.356780),
        "Las Vegas": (36.316504, -115.262798),
    }

    prj = ccrs.Orthographic(-127.305625, 47.207366)
    srcprj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=prj)

    ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='#457b9d')
    ax.add_feature(cartopy.feature.LAND, zorder=0,
                   edgecolor='black', facecolor='#edede9')
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.STATES, linestyle=':', alpha=0.2)

    # m.add_feature.preset.ocean(facecolor='#457b9d')
    # m.add_feature.preset.land(facecolor='#edede9', edgecolor='black')
    ax.set_global()
    ax.gridlines(alpha=0.1)

    for key in AOIs:
        x, y = prj.transform_point(AOIs[key][1], AOIs[key][0], srcprj)
        kw = {
            'marker': '.',
            's': 50,
            'facecolors': 'red',
            'edgecolors': 'black'
        }

        if 'Reference' in key:
            kw['marker'] = '*'
            # kw['edgecolors'] = 'black'

        ax.scatter(x, y, **kw)

        ax.annotate(key, xy=(x, y),
                    textcoords='offset points', xytext=(4, 4), fontsize=14, path_effects=[pe.withStroke(linewidth=4, foreground="white")], alpha=1)
        # arrow_args = dict(arrowstyle="-")
        # ax.annotate('',
        #             xy=(x, y),
        #             xytext=(1, 1), textcoords='axes fraction',
        #             # ha="center", va="top",
        #             # bbox=bbox_args,
        #             arrowprops=arrow_args)

        # ax.annotate('',
        #             xy=(x, y),
        #             xytext=(1, 2), textcoords='axes fraction',
        #             # ha="center", va="top",
        #             # bbox=bbox_args,
        #             arrowprops=arrow_args)

    # img = plt.imread(
    #     '/Users/rbiessel/Documents/MSThesis/figures/simon_tundra.jpg')

    # imagebox = OffsetImage(img, zoom=0.08)

    # ab = AnnotationBbox(imagebox, xy=(1, 1), xycoords='axes fraction',   bboxprops={
    #                     'edgecolor': 'none', 'alpha': 1, }, annotation_clip=False, box_alignment=(0, 0))

    # ax.add_artist(ab)

    # # # print(box)
    # fig.canvas.draw()
    # print(imagebox.get_window_extent())

    # fig.tight_layout()

    plt.savefig('/Users/rbiessel/Documents/test.png', bbox_inches='tight')
    # plt.tight_layout(pad=5.0)

    plt.show()


# sub_ax = plt.axes([0.7, 0.625, 0.2, 0.2], projection=srcprj)
# sub_ax.set_extent([-154.356780, -140.356780, 60, 70])
# sub_ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='#457b9d')
# sub_ax.add_feature(cartopy.feature.LAND, zorder=0,
#                    edgecolor='black', facecolor='#edede9')

# x, y, u, v, vector_crs = sample_data()
# ax.quiver(x, y, u, v, transform=vector_crs)
# plt.draw()
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.tight_layout()
# fig.subplots_adjust(bottom=0.2, top=1, left=0.12, right=1)
# fig.bbox_extra_artist = (ab,)


# ax2 = fig.add_axes([2, 0.1, 2.5, 0.75])


if __name__ == '__main__':
    main()
