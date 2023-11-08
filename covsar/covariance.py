import numpy as np
from numpy.core.numeric import indices
import library as sarlab
from multiprocessing import Process


class CovarianceMatrix():
    def __init__(self, stack,  ml_size=(20, 4), sample=(7, 7), landcover=None, sig=None, doprint=True, show=False):
        if doprint:
            print('Computing Covariance Matrix')
        if landcover is not None:
            print('Landcover size: ', landcover.shape)

        if stack is not None:
            slcn = stack.shape[0]

            if sample is not None:
                cov = np.zeros((slcn, slcn, stack[:, ::sample[0], :: sample[1]].shape[1],
                                stack[:, ::sample[0], :: sample[1]].shape[2]), dtype=np.complex64)
            else:
                cov = np.zeros(
                    (slcn, slcn, stack.shape[1], stack.shape[2]), dtype=np.complex64)

            total = (slcn * (slcn - 1) / 2) + slcn
            self.n = 0

        def update_cov(stack, i, j):
            self.n += 1
            cov[i, j, :, :] = sarlab.interfere(
                stack, i, j, ml=ml_size, show=show, aspect=1, scaling=1, sample=sample, cov=True, landcover=landcover, sig=sig)
            if j != i:
                cov[j, i, :, :] = cov[i, j, :, :].conj()

            if doprint:
                print(
                    f'Computing Filtered Interferograms: {int(((self.n)/total ) * 100)}%', end="\r", flush=True)
        if stack is not None:
            for i in range(slcn):
                for j in range(slcn):
                    if j >= i:
                        update_cov(stack, i, j)

            self.cov = cov

    def set_cov(self, cov):
        self.cov = cov

    def relook(self, looks, sample):

        new_cov = np.zeros((self.cov.shape[0], self.cov.shape[0], self.cov[:, :, ::sample[0], :: sample[1]].shape[2],
                            self.cov[:, :, ::sample[0], :: sample[1]].shape[3]), dtype=np.complex64)

        for i in range(self.cov.shape[0]):
            for j in range(self.cov.shape[1]):
                if j >= i:
                    relooked = sarlab.multilook(
                        self.cov[i, j], ml=looks)[::sample[0], ::sample[1]]
                    new_cov[i, j] = relooked
                    new_cov[j, i] = relooked.conj()

        self.set_cov(new_cov)

    def get_covariance(self):
        return self.cov

    def get_intensity(self):
        return 10 * np.log10(np.abs(np.diagonal(self.cov)))

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
