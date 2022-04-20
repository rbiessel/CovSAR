import numpy as np
from matplotlib import pyplot as plt
import time

from numpy.lib.twodim_base import tri


def rqi(A, v0, max=1000):
    v0 = v0 / np.linalg.norm(v0, 2)
    lmbda = (v0.T.conj() @ A @ v0)

    for k in range(max):
        B = (A - lmbda * np.eye(A.shape[0]))
        condB = 1/np.linalg.cond(B)
        if condB < 10e-5:
            # print(f'Breaking at iteration {k}')
            return v0, lmbda
        w = np.linalg.solve(B, v0)
        v0 = w / np.linalg.norm(w, 2)
        lmbda = (v0.T.conj() @ A @ v0)


eig_times = []
rqi_times = []

for i in range(1000):
    m = 10
    A = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    A = A @ A.T.conj()

    # print(np.linalg.norm(A - A.T.conj()))

    t = time.time()
    ew, ev = np.linalg.eigh(A)
    teigh = np.abs(time.time() - t)
    eig_times.append(teigh)

    # print(f'Time for eigh: {np.abs(teigh)}')

    v0 = np.random.randn(m) + 1j * np.random.randn(m)
    v0 = np.concatenate(([1], np.diag(A, 1)))
    # print(v0)
    # print(A)

    t = time.time()
    v, lmbda = rqi(A, v0)
    trqi = time.time() - t
    rqi_times.append(trqi)
    # print(f'Time for rqi: {np.abs(trqi)}')

    # print(f'Time ratio:: {trqi/teigh}')


plt.hist(eig_times, density=True, bins=100, label='eigh')
plt.hist(rqi_times, density=True,  bins=100, label='rqi')
plt.show()


def norm_v(v):
    v_ref = v * np.conj(v[0] / np.abs(v[0]))
    return v_ref
