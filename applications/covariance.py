import numpy as np
from numpy.core.numeric import indices
import library as sarlab
from multiprocessing import Process


class CovarianceMatrix():
    def __init__(self, stack,  ml_size=(20, 4), sample=(7, 7), landcover=None, sig=None):
        print('computing covariance matrix')
        if landcover is not None:
            print('Landcover size: ', landcover.shape)
        slcn = stack.shape[0]

        if sample is not None:
            cov = np.zeros((slcn, slcn, int((
                stack.shape[1] / sample[0])), int((stack.shape[2] / sample[1]))), dtype=np.complex64)

            print(cov.shape)
        else:
            cov = np.zeros(
                (slcn, slcn, stack.shape[1], stack.shape[2]), dtype=np.complex64)

        total = (slcn * (slcn - 1) / 2) + slcn
        self.n = 0

        def update_cov(stack, i, j):
            self.n += 1
            cov[i, j, :, :] = sarlab.interfere(
                stack, i, j, ml=ml_size, show=False, aspect=1, scaling=1, sample=sample, cov=True, landcover=landcover, sig=sig)
            if j != i:
                cov[j, i, :, :] = cov[i, j, :, :].conj()
            print(
                f'Computing Filtered Interferograms: {int(((self.n)/total ) * 100)}%', end="\r", flush=True)

        for i in range(slcn):
            for j in range(slcn):
                if j >= i:
                    update_cov(stack, i, j)

        self.cov = cov

    def get_covariance(self):
        return self.cov

    def get_intensity(self):
        return np.log10(np.abs(np.diagonal(self.cov)))

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
