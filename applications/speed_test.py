import numpy as np
from scipy import linalg as la
from numba import jit, njit
import time


@jit(nopython=True, parallel=True, )
def eig_all_numba(image):
    for i in range(image.shape[2]):
        for j in range(image.shape[3]):
            la.eig(image[:, :, i, j])


def eig_all_loop(image):
    for i in range(image.shape[2]):
        for j in range(image.shape[3]):
            la.eig(image[:, :, i, j])


def main():

    m = 20
    h1 = np.random.rand(m, m, 100, 100).astype(np.complex64)
    h2 = np.random.rand(m, m, 1, 1).astype(np.complex64)

    t1 = time.perf_counter()
    eig_all_numba(h2)
    t2 = time.perf_counter()
    print(t2 - t1)

    eig_all_numba(h1)
    t3 = time.perf_counter()
    print(t3 - t2)

    print('Looping')
    t1 = time.perf_counter()
    eig_all_loop(h1)
    t2 = time.perf_counter()
    print(t2 - t1)


main()
