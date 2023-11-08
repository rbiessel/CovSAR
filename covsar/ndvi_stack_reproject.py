import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import argparse
import glob
import gdal
import osr


def readInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='Glob path to the ndvi')
    args = parser.parse_args()
    return args


def main():
    inps = readInputs()
    files = glob.glob(inps.path)

    for file in files:

        input_raster = gdal.Open(file)

        with rasterio.open(file) as src:
            data = src.read()
            profile = src.profile.copy()

            # WGS84 LAT LON
            dst_crs = 'EPSG:4326'
            target_crs = osr.SpatialReference()
            target_crs.ImportFromEPSG(4326)
            gdal.Warp(file, input_raster, dstSRS=target_crs.ExportToWkt())

            # Calculate the transformation parameters and output profile
            # transform, width, height = calculate_default_transform(
            #     src.crs, dst_crs, src.width, src.height, *src.bounds)

            # print(src.crs)
            # profile.update({
            #     'crs': dst_crs,
            #     'transform': transform,
            #     'width': width,
            #     'height': height
            # })

            # with rasterio.open(file, 'w', **profile) as dst:
            #     reproject(
            #         source=rasterio.band(src, 1),
            #         destination=rasterio.band(dst, 1),
            #         src_transform=src.transform,
            #         src_crs=src.crs,
            #         dst_transform=transform,
            #         dst_crs=dst_crs,
            #         resampling=Resampling.nearest)

            # dst.write(data)


main()
