import numpy as np
from mintpy.geocode import run_geocode as geocode
import glob
from datetime import datetime as dt
import pandas as pd
import rasterio as rio
from matplotlib import pyplot as plt
import os


def main():
    ndvi_folder = '/Users/rbiessel/Documents/vegas_ndvi'

    paths = glob.glob(ndvi_folder + '/*.tif')
    sar_paths = glob.glob(
        '/Volumes/SARDRIVE/vegas_2020/corncreek/SLC/**/*.slc.full')

    dates = [path.split('_')[3].replace('.tif', '') for path in paths]
    dates = [dt.strptime(date, '%Y%m%d') for date in dates]

    sar_dates = [path.split('/')[-2] for path in sar_paths]
    sar_dates = [dt.strptime(date, '%Y%m%d') for date in sar_dates]

    # indices = np.argsort(dates)
    # dates = dates[indices]
    # paths = paths[indices]

    # print(sar_dates)

    raster_data = []

    # Loop through each raster file and load the data into a numpy array
    shape = None
    rio_profile = None

    for raster_file in paths:
        with rio.open(raster_file) as src:
            array = src.read(1)
            if shape is None:
                shape = array.shape
            if rio_profile is None:
                rio_profile = src.profile

                rio_profile.update(
                    dtype=rio.float32,
                    nodata=-9999,
                    count=1)

            array = array.reshape(shape[0] * shape[1])
            raster_data.append(array)

    df = pd.DataFrame(data=np.array(raster_data), index=dates)
    df = df.sort_index()
    # df = df.groupby(level=0).mean()
    # # print(df)

    # resampled_df = df.resample('12D').mean()
    # reindexed_df = df.reindex(sar_dates + dates, fill_value=None)
    new_index = pd.date_range(start=df.index.min(),
                              end=df.index.max(), freq='D')
    new_df = df.reindex(new_index)
    new_df = new_df.interpolate(method='cubic')
    print(df)

    # sarray = new_df.loc[sar_dates[5]].to_numpy().reshape(shape)

    out_folder = '/Users/rbiessel/Documents/vegas_ndvi/resampled/'
    for date in sar_dates:
        sarray = new_df.loc[date].to_numpy().reshape(shape)
        plt.imshow(sarray, vmin=0, vmax=0.5)
        opath = os.path.join(out_folder, date.strftime('%Y%m%d') + '.tif')
        print(opath)

        with rio.open(opath, 'w', **rio_profile) as dst:
            dst.write(sarray, 1)
    return
    latFile = 'test'
    lonFile = 'test'

    'geocode.py filt_fine.int --lat-file ../../geom_reference/lat.rdr --lon-file ../../geom_reference/lon.rdr'

    inpts = {
        file: ['files'],
        lat-file: '',
        lon-file: '',

    }

    geocode()

    for path in paths:
        print(path)
        # run geocode


main()
