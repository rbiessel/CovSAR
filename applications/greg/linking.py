'''
Created on Oct 25, 2021

@author: simon
'''
import numpy as np
from scipy.linalg import eigh
from numba import njit
from greg.preproc import (
    correlation, force_doubly_nonnegative, force_doubly_nonnegative_py)


def EMI(C_obs, G=None, corr=True):
    # output magnitude not normalized
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P:
        raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        G = force_doubly_nonnegative(np.abs(C_obs).real, inplace=True)
    G = G.reshape((-1, P, P))
    if corr:
        G = correlation(G, inplace=False)
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.array(_EMI(C_obs, G))
    ceig *= (ceig[:, 0].conj() / np.abs(ceig[:, 0]))[:, np.newaxis]
    return ceig.reshape(C_shape[:-1])


def EMI_py_stack(C_obs, G=None, corr=True):
    ts = np.zeros((C_obs.shape[2], C_obs.shape[3],
                   C_obs.shape[0]), dtype=np.complex64)

    for i in range(C_obs.shape[2]):
        for j in range(C_obs.shape[3]):
            ts[i, j, :] = EMI_py(C_obs[:, :, i, j], G=None, corr=corr)

    return ts


def EMI_py(C_obs, G=None, corr=True):
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P:
        raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        G = force_doubly_nonnegative_py(np.abs(C_obs).real)
    G = G.reshape((-1, P, P))
    N = G.shape[0]
    if corr:
        G = correlation(G, inplace=False)
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.empty((N, P), dtype=np.complex128)
    for n in range(N):
        had = C_obs[n, :, :]
        had *= np.linalg.pinv(G[n, :, :])
        _, ceig_n = eigh(
            had, subset_by_index=[0, 0], eigvals_only=False)
        ceig[n, :] = ceig_n[:, 0] * \
            (ceig_n[0, 0].conj() / np.abs(ceig_n[0, 0]))
    return ceig.reshape(C_shape[:-1])


if __name__ == '__main__':
    C_obs = np.array([[3, 1j], [-1j, 4]])
    ceig = EMI_py(C_obs)
