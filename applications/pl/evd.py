import numpy as np


def eig_decomp(cov):
    cov = np.swapaxes(cov, 0, 2)
    cov = np.swapaxes(cov, 1, 3)
    shape = cov.shape
    cov = cov.reshape(
        (shape[0] * shape[1], shape[2], shape[3]))

    total_pixels = cov.shape[0]
    n = 50
    cov_split = np.array_split(cov, n)

    cov = None

    print('Eigenvector Decomposition Progress: 0', end="\r", flush=True)
    for minicov in cov_split:
        W, V = np.linalg.eigh(minicov)
        W = None
        V = np.transpose(V, axes=(0, 2, 1))
        # select the last eigenvector (the one corresponding to the largest lambda)
        v = V[:, shape[-1] - 1]
        V = None
        scaling = np.abs(v[:, 0])
        scaling[scaling == 0] = 1
        rotation = v[:, 0] / scaling
        scaling = None
        v = v * rotation.conj()[:, np.newaxis]
        if cov is None:
            cov = v
        else:
            cov = np.concatenate((cov, v))
        print(
            f'Eigenvector Decomposition Progress: {int((cov.shape[0] / total_pixels) * 100)}%', end="\r", flush=True)

    return cov.reshape((shape[0], shape[1], shape[2]))
