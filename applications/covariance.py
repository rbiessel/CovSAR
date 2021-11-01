import numpy as np
from numpy.core.numeric import indices
import library as sarlab


class CovarianceMatrix():
    def __init__(self, stack,  ml_size=(20, 4), landcover=None, sig=None):
        if landcover is not None:
            print('Landcover size: ', landcover.shape)
        slcn = stack.shape[0]
        cov = np.zeros(
            (slcn, slcn, stack.shape[1], stack.shape[2]), dtype=np.complex64)

        n = 0
        total = (slcn * (slcn - 1) / 2) + slcn

        for i in range(slcn):
            for j in range(slcn):
                if j >= i:
                    n += 1
                    cov[i, j, :, :] = sarlab.interfere(
                        stack, i, j, ml=ml_size, show=False, aspect=1, scaling=1, cov=True, landcover=landcover, sig=sig)
                    print(
                        f'Computing Filtered Interferograms: {int(((n)/total ) * 100)}%', end="\r", flush=True)
                else:
                    cov[i, j, :, :] = cov[j, i, :, :].conj()

        print()
        self.cov = cov

    def get_covariance(self):
        return self.cov

    def get_coherence(self):
        coherence = np.zeros(self.cov.shape, dtype=np.complex64)
        print('shape: ', coherence.shape)
        for i in range(coherence.shape[0]):
            for j in range(coherence.shape[1]):
                if j >= i:
                    coherence[i, j] = self.cov[i, j] / \
                        np.sqrt((self.cov[i, i] * self.cov[j, j]))
                else:
                    coherence[i, j] = coherence[j, i].conj()
        print('Finished Computing Coherence')
        return coherence

    def get_intensity(self):
        return np.diag(self.cov)
