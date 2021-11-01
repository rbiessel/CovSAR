#!/usr/bin/env python3

# Author: Heresh Fattahi
# Modified by Rowan Biessel

import os
import argparse
from osgeo import gdal
import isce
import isceobj


def cmdLineParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='unwrap estimated wrapped phase',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--interferogram_file', type=str, dest='interferogramFile',
                        required=True, help='Input interferogram file with complex format')

    parser.add_argument('-c', '--coherence_file', type=str, dest='coherenceFile',
                        required=True, help='Input coherence file')

    parser.add_argument('-o', '--unwrap_file', type=str, dest='unwrapFile',
                        required=True, help='Output unwrapped file')

    parser.add_argument('-m', '--method', type=str, dest='method',
                        default='snaphu', help='unwrapping method: default = snaphu')

    parser.add_argument('-x', '--xml_file', type=str, dest='xmlFile',
                        required=False, help='path of reference xml file for unwrapping with snaphu')

    return parser.parse_args()


# Adapted code from unwrap.py and s1a_isce_utils.py in topsStack
def extractInfo(inps):
    '''
    Extract required information from pickle file.
    '''
    from isceobj.Planet.Planet import Planet
    from isceobj.Util.geo.ellipsoid import Ellipsoid
    from iscesys.Component.ProductManager import ProductManager as PM

    pm = PM()
    frame = pm.loadProduct(inps.xmlFile)

    burst = frame.bursts[0]
    planet = Planet(pname='Earth')
    elp = Ellipsoid(planet.ellipsoid.a, planet.ellipsoid.e2, 'WGS84')

    data = {}
    data['wavelength'] = burst.radarWavelength

    tstart = frame.bursts[0].sensingStart
    #tend   = frame.bursts[-1].sensingStop
    #tmid = tstart + 0.5*(tend - tstart)
    tmid = tstart

    orbit = burst.orbit
    peg = orbit.interpolateOrbit(tmid, method='hermite')

    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    hdg = orbit.getENUHeading(tmid)
    refElp.setSCH(llh[0], llh[1], hdg)

    earthRadius = refElp.pegRadCur

    altitude = llh[2]

    data['altitude'] = altitude  # llh.hgt

    data['earthRadius'] = earthRadius
    return data


def unwrap_snaphu(interferogramFile, coherenceFile, unwrapFile, metadata, range_looks: int, azimuth_looks: int):
    length, width = getSize(interferogramFile)

    from contrib.Snaphu.Snaphu import Snaphu

    if metadata is None:
        altitude = 800000.0
        earthRadius = 6371000.0
        wavelength = 0.056
    else:
        altitude = metadata['altitude']
        earthRadius = metadata['earthRadius']
        wavelength = metadata['wavelength']

    snp = Snaphu()
    snp.setInitOnly(False)
    snp.setInput(interferogramFile)
    snp.setOutput(unwrapFile)
    snp.setWidth(width)
    snp.setCostMode('DEFO')
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(coherenceFile)
    snp.setInitMethod('MST')
   # snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(100)
    snp.setDefoMaxCycles(.0)
    snp.setRangeLooks(range_looks)
    snp.setAzimuthLooks(azimuth_looks)
    snp.setIntFileFormat('FLOAT_DATA')
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    write_xml(unwrapFile, width, length, 2, "FLOAT", "BIL")
    write_xml(unwrapFile+'.conncomp', width, length, 1, "BYTE", "BIP")


def write_xml(fileName, width, length, bands, dataType, scheme):

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()

    return None


def getSize(data):
    gdal.AllRegister()
    ds = gdal.Open(data + '.vrt', gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize
    # print(ds.DataType)
    ds = None
    return length, width


if __name__ == '__main__':
    '''
    Main driver.
    '''
    #*************************************************************#
    # read the input options and unwrap
    inps = cmdLineParser()
    print("length, width: ", length, " ", width)

    unwrapDir = os.path.dirname(inps.unwrapFile)
    if not os.path.exists(unwrapDir):
        os.makedirs(unwrapDir)

    if inps.method == "snaphu":
        if inps.xmlFile is not None:
            metadata = extractInfo(inps)
        else:
            metadata = None
        unwrap_snaphu(inps.interferogramFile, inps.coherenceFile,
                      inps.unwrapFile, metadata)
