# CovSAR

Covariance Based Full-Network InSAR Time Series Extraction from an ISCE stack of SLCs with Python.

The core script "./applications/closure_inten.py" will compute a phase history and phase history corrected with an intensity triplet.

For use with UAVSAR data, the file "./applications/subset_uavsar.py" converts a stack of UAVSAR SLCs into a format that's compatible with a Sentinel-1 stack.

Please note that this repository is experimental code I have not optimized it easy use or speed.

## Requirements

I have also yet to compile a list of dependencies, but core requirements are ISCE (https://github.com/isce-framework/isce2) and GREG (https://github.com/szwieback/greg).

## Contact

Contact me at rbiessel@alaska.edu

## Citation

R. Biessel and S. Zwieback, “CovSAR.” Jul. 22, 2022. Available: https://github.com/rbiessel/CovSAR
