import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import cartopy as cartopy
import figStyle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from cartopy.io.img_tiles import Stamen
tiler = Stamen('terrain-background')
mercator = tiler.crs


df = pd.read_csv('/Users/rbiessel/Documents/InSAR/vegas_weather/2021.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
stations = set(list(df['STATION']))

if True:
    lats = df['LATITUDE'].values
    lons = df['LONGITUDE'].values
    names = df['NAME'].values

    indexes = np.unique(lats, return_index=True)[1]

    lats = lats[indexes]
    lons = lons[indexes]
    names = names[indexes]

    extent = [np.min(lons) - 0.1, np.max(lons)+0.1,
              np.min(lats)-0.1, np.max(lats)+0.1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=mercator)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_image(tiler, 12)
    ax.scatter(x=lons, y=lats, transform=ccrs.PlateCarree(),
               color='black', alpha=0.8)

    for i in range(len(lons)):
        ax.annotate(names[i], xy=(lons[i], lats[i]), xycoords=ccrs.PlateCarree(),
                    ha='right', va='top')
    plt.show()


for station in stations:
    station_df = df.loc[df['STATION'] == station]
    if True:  # 'DESERT' in station_df['NAME'].values[0]:

        dates = station_df['DATE']
        print(station)

        print('Latitude: ', station_df['LATITUDE'].values[0])
        print('Longitude: ', station_df['LONGITUDE'].values[0])

        # plt.plot(station_df['DATE'], station_df['PRCP'],
        #          label=r'Precipitation [$mm$]')
        plt.plot(station_df['DATE'], station_df['TMIN'],
                 label=r'Minimum Temperature [$C^{\circ}$]')
        plt.plot(station_df['DATE'], station_df['TMAX'],
                 label=r'Max Temperature [$C^{\circ}$]')
        plt.plot(station_df['DATE'], station_df['SNOW'], '--',
                 label=r'Snow [$mm$]')
        plt.plot(station_df['DATE'], station_df['SNWD'], '--',
                 label=r'Snow [$mm$]')
        plt.xlabel('Date')
        plt.legend(loc='best')

        plt.title(
            f"{station_df['NAME'].values[0]} at {station_df['ELEVATION'].values[0]} m")
        plt.show()
