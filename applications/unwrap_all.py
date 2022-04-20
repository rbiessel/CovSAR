
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
    parser.add_argument('-al', '--az-looks', type=int, dest='alooks',
                        required=True, help='number of azimuth looks')

    parser.add_argument('-rl', '--range-looks', type=int, dest='rlooks',
                        required=True, help='number of range looks')

    parser.add_argument('-c', '--coherence_file', type=str, dest='coherenceFile',
                        required=True, help='Input coherence file')

    return parser.parse_args()


def unwrap_interferogram(int_path, coh_path, rlooks, alooks):
    int_path = int_path.replace('unwrapped', 'wrapped')
    out_path = int_path.replace('wrapped', 'unwrapped')
    unwrap.unwrap_snaphu(int_path, coherenceFile=coh_path, unwrapFile=out_path,
                         metadata=None, range_looks=rlooks, azimuth_looks=alooks)


def main():
    inputs = cmdLineParser()
    int_files = sorted(glob.glob(inputs.interferogramFiles))
    for i in range(len(int_files)):
        p = Process(target=unwrap_interferogram, args=(
            int_files[i], inputs.coherenceFile, inputs.rlooks, inputs.alooks))
        p.start()


main()
