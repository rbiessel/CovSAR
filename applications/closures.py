import numpy as np
import isceio as io
import itertools
import os
import shutil
import library as sarlab
from scipy import stats, special
from scipy.stats import chi2
from matplotlib import pyplot as plt


def get_adjacent_triplets(num):

    triplets = []

    for i in range(num - 2):
        triplets.append([i, i + 1, i + 2])

    return np.sort(np.array(triplets))


def get_triplets(num):

    numbers = np.arange(0, num, 1)
    # print(numbers)

    permutations = np.array(list(itertools.permutations(numbers, 3)))
    permutations = np.sort(permutations, axis=1)
    permutations = np.unique(permutations, axis=1)

    return permutations


def get_triplet_covariance(cov):
    phi_cov = get_phase_covariance(cov, count=(30**2))
    print(phi_cov.shape)
    phi_cov = np.swapaxes(phi_cov, 0, 2)
    phi_cov = np.swapaxes(phi_cov, 1, 3)
    df = int(special.comb(cov.shape[0] - 1, 2))
    A = np.array([[1, 1, -1], ] * df)
    triplet_covariance = A @ phi_cov @ A.T
    return triplet_covariance


def get_closure_significance(triplet, triplet_variance):
    k = special.comb(2, 2)
    print(f'{k} degrees of freedom')
    triplet_variance = np.squeeze(triplet_variance)
    # return 1, 1
    S = np.angle(triplet) * triplet_variance**-1 * np.angle(triplet)

    return 1 - chi2.cdf(S, df=k), S


def write_closures(coherence, folder):
    trip_covar = get_triplet_covariance(coherence)

    triplets = get_adjacent_triplets(coherence.shape[0])

    folder_path = os.path.join(
        '/Users/rbiessel/Documents/InSAR/Toolik/Fringe', folder)
    if os.path.exists(folder_path):
        print('Output folder already exists, clearing it')
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    for triplet in triplets:

        triplet = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        pval, s = get_closure_significance(triplet, trip_covar)
        pval = pval.astype(np.float32)
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(pval)
        ax[1].imshow(np.angle(triplet))
        ax[2].imshow(np.log10(np.abs(triplet)))
        plt.show()
        outpath = os.path.join(
            folder_path, f'pval.closure')
        io.write_image(outpath, pval)
        closure_out = os.path.join(
            folder_path, f'x123.closure')
        io.write_image(closure_out, np.angle(triplet))


def get_phase_covariance(cov, count=0):
    def get_phi_cov(cov, i, j, k, l):
        '''
        Compute the covariance between two interferograms:
        cov - The first order covariance of the SLCs
        i,j - the indexes of the first interogram
        k,l - the indexes of the second interferogram
        '''

        # if i == k and j == l:
        #     print('Computing the variance for the diagonal')
        #     norm = 1 / (2 * size**2)
        #     return norm * ((1 - np.abs(cov[i, j])**2) / (np.abs(cov[i, j])**2))

        norm = (1 / (2 * (count/2) *
                     np.abs(cov[i, j]) * np.abs(cov[k, l])))

        c1 = np.angle(cov[i, j] * cov[j, k] * cov[i, k].conj())
        c2 = np.angle(cov[i, j] * cov[j, l] * cov[i, l].conj())
        c3 = np.angle(cov[j, k] * cov[k, l] * cov[j, l].conj())

        t1 = np.abs(cov[i, k]) * np.abs(cov[j, l]) * np.cos(c1 - c3)
        t2 = np.abs(cov[i, l]) * np.abs(cov[j, k]) * np.cos(c2 + c3)

        return ((t1 - t2) * norm).astype(np.float64)

    index_matrix = []
    for i in range(cov.shape[0]):
        row = []
        for j in range(cov.shape[1]):
            row.append(f'{i}{j}')
        index_matrix.append(row)

    cov_indexes = np.array(index_matrix)
    cov_indexes = cov_indexes[np.triu_indices(cov_indexes.shape[0], 1)]
    print(cov_indexes)
    intn = len(cov_indexes)
    phi_cov = np.zeros((intn, intn, cov.shape[2], cov.shape[3]))

    for a in range(cov_indexes.shape[0]):
        for b in range(cov_indexes.shape[0]):
            i = int(cov_indexes[a][0])
            j = int(cov_indexes[a][1])

            k = int(cov_indexes[b][0])
            l = int(cov_indexes[b][1])
            phi_cov[a, b] = get_phi_cov(cov, i, j, k, l)

    return phi_cov
