from audioop import mul
from email import header
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
# from geojson import Point, MultiPoint

to_export = []


def main():

    folder_path = '/Users/rbiessel/Documents/smdata/oklahoma_smap_val'
    files = glob.glob(folder_path + '/**/*.txt')

    points = []

    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    for file in files:
        with open(file) as f:
            first_line = f.readline()

        lat = float(first_line.split('lat: ')[1].split(',')[0])
        lon = float(first_line.split('lon: ')[1].split(')')[0])
        print(f'Latitude: {lat}, Longitude: {lon}')
        point = (lon, lat)
        # plt.scatter(lon, lat)
        df = pd.read_csv(file, sep=",", header=1)

        df.columns = df.columns.str.replace(' ', '')

        years = df['Yr'].str.strip().values[1:]
        months = df['Mo'].str.strip().values[1:]
        days = df['Day'].str.strip().values[1:]
        hours = df['Hr'].values[1:].astype(str)
        mins = df['Min'].str.strip().values[1:]

        dates = (years + '-' + months + '-' + days +
                 'T' + hours + ':' + mins)
        dates = dates[np.where(dates != '--')]
        dates = pd.to_datetime(dates)

        WASM = df['SM-1'].values[1:].astype(np.float32)

        valid_mask = np.where(WASM > 0)
        dates = dates[valid_mask]
        WASM = WASM[valid_mask]
        label = file.split('_')[-3]
        if point not in points:
            if lat <= 35:

                data = {'Dates': dates, 'WASM': WASM}
                newdf = pd.DataFrame(data=data)
                newdf.attrs['lat'] = lat
                newdf.attrs['lon'] = lon
                to_export.append(newdf)

                # ax[1].set_title('Southern Oklahoma')
                # ax[1].plot(dates, WASM, label=label)
                points.append(point)


def get_sm_ts():
    return to_export


main()
