import numpy as np
import library as sarlab


def eval_triplets(triplets, cov_matrix, filter_strength=1):

    cov = cov_matrix.cov
    closure_stack = np.zeros((
        len(triplets), cov.shape[2], cov.shape[3]), dtype=np.complex64)

    amp_triplet_stack = closure_stack.copy()
    amp_triplet_stack = amp_triplet_stack.astype(np.float64)

    intensity = cov_matrix.get_intensity()

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = cov[triplet[0], triplet[1]] * cov[triplet[1],
                                                    triplet[2]] * cov[triplet[0], triplet[2]].conj()

        closure = sarlab.multilook(closure, ml=(
            filter_strength, filter_strength))

        amp_triplet = sarlab.intensity_closure(
            intensity[:, :, triplet[0]], intensity[:, :, triplet[1]], intensity[:, :, triplet[2]], norm=False, cubic=False, filter=1, kappa=1)

        closure_stack[i] = closure
        amp_triplet_stack[i] = amp_triplet

    closure_stack[np.isnan(closure_stack)] = 0
    amp_triplet_stack[np.isnan(amp_triplet_stack)] = 0

    return closure_stack, amp_triplet_stack
