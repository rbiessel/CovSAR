import numpy as np
from covariance import CovarianceMatrix
from scipy import stats
import closures
from triplets import eval_triplets
from greg import simulation as greg_sim
import library as sarlab
from matplotlib import pyplot as plt


def get_random_C(C, l, coherence=False):
    sim_data = greg_sim.circular_normal(
        (l, l, C.shape[0]), Sigma=C)
    # evaluate covariance
    C_sim = C.copy()
    for i in range(C.shape[0]):
        for j in range(i, C.shape[1]):
            C_sim[i, j] = np.mean(sim_data[:, :, i] *
                                  sim_data[:, :, j].conj())
            C_sim[j, i] = C_sim[i, j].conj()

    # Normalize
    if coherence:
        for i in range(C.shape[0]):
            for j in range(i, C.shape[1]):
                C_sim[i, j] = C_sim[i, j] / np.sqrt(C_sim[i, i] * C_sim[j, j])

    return C_sim


def bootstrap_correlation(C, l, triplets, nsample=100, fitLine=False, zeroPhi=False):

    r2 = np.zeros((nsample))

    if fitLine:
        coeffs = np.zeros((2, nsample))

    if zeroPhi:
        C = C * C.conj() / np.abs(C)

    for i in range(nsample):

        sim_data = greg_sim.circular_normal(
            (l, l, C.shape[0]), Sigma=C)

        sim_data = np.swapaxes(sim_data, 0, 2)
        sim_data = np.swapaxes(sim_data, 1, 2)

        # sim_cov = CovarianceMatrix(
        #     sim_data, ml_size=(l, l), sample=(l, l), doprint=False)
        # print(sim_cov.cov.shape)

        sim_cov = CovarianceMatrix(stack=None, doprint=False)
        cov = get_random_C(C, l, coherence=False)[:, :, np.newaxis, np.newaxis]
        # print(cov.shape)
        sim_cov.set_cov(cov)

        sim_closure_stack, sim_amp_triplets = eval_triplets(
            triplets, sim_cov)

        rss, pss = stats.pearsonr(
            sim_amp_triplets.flatten(), np.angle(sim_closure_stack).flatten())

        r2[i] = rss

        if fitLine:
            coeff, covm = sarlab.gen_lstq(sim_amp_triplets.flatten(), np.angle(sim_closure_stack).flatten(
            ), W=None, function='linear')
            coeffs[:, i] = coeff

    if fitLine:
        return r2, coeffs
    else:
        return r2
