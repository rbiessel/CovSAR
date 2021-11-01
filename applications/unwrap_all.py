
import os
import argparse
from osgeo import gdal
import isce
import isceobj
import unwrap
import glob
from multiprocessing import Process


def cmdLineParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='unwrap estimated wrapped phase',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='interferogramFiles',
                        required=True, help='Folder containing folders with')
    parser.add_argument('-l', '--looks', type=int, dest='looks',
                        required=True, help='number of looks')

    parser.add_argument('-c', '--coherence_file', type=str, dest='coherenceFile',
                        required=True, help='Input coherence file')

    return parser.parse_args()


def unwrap_interferogram(int_path, coh_path):
    int_path = int_path.replace('unwrapped', 'wrapped')
    out_path = int_path.replace('wrapped', 'unwrapped')
    unwrap.unwrap_snaphu(int_path, coherenceFile=coh_path, unwrapFile=out_path,
                         metadata=None, range_looks=20, azimuth_looks=20)


def main():
    inputs = cmdLineParser()
    int_files = sorted(glob.glob(inputs.interferogramFiles))
    for i in range(len(int_files)):
        p = Process(target=unwrap_interferogram, args=(
            int_files[i], inputs.coherenceFile))
        p.start()


main()
