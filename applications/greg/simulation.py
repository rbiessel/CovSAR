'''
Created on Oct 25, 2021

@author: simon
'''
import numpy as np
from scipy.linalg import cholesky, toeplitz

import pyximport; pyximport.install()


# simulate white noise
def circular_white_normal(size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    y = rng.standard_normal(size=size) + 1j * rng.standard_normal(size=size)
    return y / np.sqrt(2)


# simulate Gaussian speckle with general Sigma
def circular_normal(size, Sigma=None, rng=None, cphases=None):
    y = circular_white_normal(size, rng=rng)
    if Sigma is not None:
        assert Sigma.shape[0] == Sigma.shape[1]
        assert len(Sigma.shape) == 2
        assert size[-1] == Sigma.shape[0]
        L = cholesky(Sigma, lower=True)
        y = np.einsum('ij,...j->...i', L, y, optimize=True)
        if cphases is not None:
            y = np.einsum('j, ...j->...j', cphases, y, optimize=True)
    return y


def decay_model(
        R=500, L=100, P=40, coh_decay=0.9, coh_infty=0.1, incoh_bad=None,
        cphases=None, rng=None):
    # using the Cao et al. phase convention
    Sigma = ((1 - coh_infty) * toeplitz(np.power(coh_decay, np.arange(P)))
             +coh_infty * np.ones((P, P))).astype(np.complex128)
    if incoh_bad is not None:
        intensity = Sigma[P // 2, P // 2]
        Sigma[P // 2,:] *= incoh_bad
        Sigma[:, P // 2] *= incoh_bad
        Sigma[P // 2, P // 2] = intensity
    y = circular_normal((R, L, P), Sigma=Sigma, rng=rng, cphases=cphases)
    return y

# if __name__ == '__main__':
#     rng = np.random.default_rng()
#     Sigma = np.array([[5, 1j], [-1j, 3]]).astype(np.complex128)
#     y = decay_model(100, incoh_bad=0.5)
#     C_obs = covariance_matrix(y)

