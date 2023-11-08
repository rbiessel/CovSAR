#!python
# cython: language_level=3

import numpy as np
# from scipy.linalg import eigh
from scipy.linalg import eigh

cimport cython

cdef extern from "complex.h":
    double complex conj(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def covm(double complex[:, :, :] y):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig

    cdef Py_ssize_t R = y.shape[0]
    cdef Py_ssize_t L = y.shape[1]
    cdef Py_ssize_t P = y.shape[2]

    cdef Py_ssize_t n, p1, p2, l

    C = np.zeros((R, P, P), dtype=np.complex128)
    cdef double complex [:, :, :] C_view = C
    Cr = np.zeros((P, P), dtype=np.complex128)
    cdef double complex [:, :] Cr_view = Cr


    for r in range(R):
        Cr_view[:, :] = 0.0
        for l in range(L):
            for p1 in range(P):
                for p2 in range(p1, P):
                    Cr_view[p1, p2] = Cr_view[p1, p2] + y[r, l, p1] * conj(y[r, l, p2])
        for p1 in range(P):
            for p2 in range(P):
                if p2 >= p1:
                    C_view[r, p1, p2] = Cr_view[p1, p2] / L
                else:
                    C_view[r, p1, p2] = conj(Cr_view[p2, p1]) / L
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
def fdd(double[:, :, :] G_raw, double[:, :, :] G_out, double min_eig=0.0):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig

    assert G_raw.shape[1] == G_raw.shape[2]
    cdef Py_ssize_t N = G_raw.shape[0]
    cdef Py_ssize_t P = G_raw.shape[1]

    cdef Py_ssize_t n, p

    lam = np.ones(1, dtype=np.double)
    cdef double [:] lam_view = lam

    for n in range(N):
        # may need to call lapack directly
        lam_view[:] = eigh(
            G_raw[n, :, :], subset_by_index=[0, 0], eigvals_only=True)[0]

        G_out[n, :, :] = G_raw[n, :, :]
        negshift = lam_view[0] - min_eig
        if negshift < 0:
            for p in range(P):
                G_out[n, p, p] -= negshift
    return G_out

@cython.boundscheck(False)
@cython.wraparound(False)
def _EMI(double complex[:, :, :] C_obs, double[:, :, :] G):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig
 
    assert G.shape[1] == G.shape[2]
#     assert C_obs.shape == G.shape
    cdef Py_ssize_t N = G.shape[0]
    cdef Py_ssize_t P = G.shape[1]
 
    cdef Py_ssize_t n, p1, p2
 
    ceig = np.empty((N, P), dtype=np.complex128)
    cdef double complex [:, :] ceig_view = ceig
    ceig_n = np.empty((P, 1), dtype=np.complex128)
    cdef double complex [:, :] ceig_n_view = ceig_n
    M1 = np.empty((P, P), dtype=np.complex128)
    M2 = np.empty((P, P), dtype=np.complex128)
    cdef double complex [:, :] M2_view = M2
    lam = np.ones(1, dtype=np.double)
    cdef double [:] lam_view = lam
    
    for n in range(N):
        # may need to call lapack directly
        M1[:, :] = np.linalg.pinv(G[n, :, :])
        M2[:, :] = C_obs[n, :, :]            
        M1 *= M2
        # may need to call lapack directly
        lam, ceig_n_view = eigh(M1, subset_by_index=[0, 0], eigvals_only=False)
        ceig_view[n, :] = ceig_n_view[:, 0]
    return ceig_view
