import numpy as np
from covariance import CovarianceMatrix
from scipy import stats
import closures
from triplets import eval_triplets
from greg import simulation as greg_sim
import library as sarlab


def bootstrap_correlation(C, l, triplets, nsample=100, fitLine=False, zeroPhi=False):

    r2 = np.zeros((nsample))

    if fitLine:
        coeffs = np.zeros((2, nsample))

    if zeroPhi:
        C = C * C.conj()

    for i in range(nsample):

        sim_data = greg_sim.circular_normal(
            (l, l, C.shape[0]), Sigma=C)

        sim_data = np.swapaxes(sim_data, 0, 2)
        sim_data = np.swapaxes(sim_data, 1, 2)

        sim_cov = CovarianceMatrix(
            sim_data, ml_size=(l, l), sample=(l, l), doprint=False)

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
