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


def get_triplets(num, force=None):
    numbers = np.arange(0, num, 1)
    permutations = np.array(list(itertools.permutations(numbers, 3)))
    combinations = np.sort(permutations, axis=1)
    combinations = np.unique(combinations, axis=0)
    if force is not None:
        combinations = np.array(
            [triplet for triplet in combinations if triplet[0] == force])

    return combinations


def build_A(triplets, phi_indexes):
    A = np.zeros((triplets.shape[0], len(phi_indexes)))
    for i in range(triplets.shape[0]):
        a = A[i]
        triplet = triplets[i]
        a[np.where(phi_indexes == f'{triplet[0]}{triplet[1]}')] = 1
        a[np.where(phi_indexes == f'{triplet[0]}{triplet[2]}')] = -1
        a[np.where(phi_indexes == f'{triplet[1]}{triplet[2]}')] = 1
    print('Rank:', np.linalg.matrix_rank(A), 'Shape: ', A.shape)
    return A


def get_triplet_covariance(cov, triplets):
    phi_cov, indexes = get_phase_covariance(
        cov, count=(30**2))  # dont hard-code the sample size
    A = build_A(triplets, indexes)
    phi_cov = np.swapaxes(phi_cov, 0, 2)
    phi_cov = np.swapaxes(phi_cov, 1, 3)
    triplet_covariance = A @ phi_cov @ A.T
    return triplet_covariance, indexes, A


def get_closure_significance(closures, triplet_covariance, N):
    k = special.comb(N-1, 2)
    print(f'{k} degrees of freedom')
    triplet_covariance = np.squeeze(triplet_covariance)

    # Does this need to be the full Sigma_xi matrix inverse?

    closures = closures[:, :, :, np.newaxis]
    print(closures.shape)
    print(triplet_covariance.shape)
    S = np.transpose(closures, (0, 1, 3, 2)
                     ) @ np.linalg.inv(triplet_covariance) @ closures

    return 1 - chi2.cdf(S, df=k), S


def write_closures(coherence, folder):
    N = coherence.shape[0]
    # triplets = get_triplets(coherence.shape[0], force=0)
    triplets = get_adjacent_triplets(coherence.shape[0])
    trip_covar, indexes, A = get_triplet_covariance(coherence, triplets)
    coherence = np.angle(coherence).astype(np.float32)

    indexes = np.triu_indices_from(coherence[:, :, 1, 1], k=1)
    coherence = coherence[indexes]

    coherence = np.swapaxes(coherence, 0, 1)
    coherence = np.swapaxes(coherence, 2, 1)

    closures = np.squeeze(A @ coherence[:, :, :, np.newaxis])

    closures = (closures + np.pi) % (2 * np.pi) - np.pi
    # plt.imshow(closures[:, :, 1])
    # plt.show()

    pval, S = get_closure_significance(closures, trip_covar, N=N)
    pval = np.squeeze(pval).astype(np.float32)

    folder_path = os.path.join(
        '/Users/rbiessel/Documents/InSAR/Toolik/Fringe', folder)
    if os.path.exists(folder_path):
        print('Output folder already exists, clearing it')
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    print(closures.shape)

    pval_out = os.path.join(folder_path, 'pval.closure')
    io.write_image(pval_out, pval)

    for i in range(triplets.shape[0]):
        triplet = triplets[i]
        closure_out = os.path.join(
            folder_path, f'Xi_{triplet[0]}{triplet[1]}{triplet[2]}.closure')
        print(closures.dtype)
        plt.imshow(closures[:, :, i])
        plt.show()
        io.write_image(closure_out, closures[:, :, i].astype(np.float32))


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
        print('Computed Phase Variance')
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

    return phi_cov, cov_indexes
