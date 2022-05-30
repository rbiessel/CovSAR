import numpy as np
from matplotlib import pyplot as plt


def nearest_neighbor(cov):
    nn = np.cumprod(np.diagonal(cov, 1), axis=2)
    zeroi = np.ones((cov.shape[2], cov.shape[3]), dtype=np.complex64)
    nn = np.insert(nn, 0, zeroi, axis=2)
    return nn
