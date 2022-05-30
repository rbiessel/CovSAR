import numpy as np
import isceio as io
import itertools
import os
import shutil
import library as sarlab
from scipy import stats, special
from scipy.stats import chi2
from matplotlib import pyplot as plt


def eval_sytstematic_closure(amptriplets, model, form='linear'):
    '''
        Evaluate amplitude triplets based on model parameters in either a linear or cubic root form
    '''
    if form is 'linear':
        est_closures = model[0] * amptriplets.flatten()
    if form is 'lineari':
        est_closures = model[0] * amptriplets.flatten() + model[1]
    elif form is 'root3':
        est_closures = model[0] * np.sign(amptriplets) * np.abs(
            amptriplets.flatten())**(1/3) + model[1] * amptriplets.flatten()
    elif form is 'root5':
        est_closures = model[0] * np.sign(amptriplets) * np.abs(
            amptriplets.flatten())**(1/5) + model[1] * np.sign(amptriplets) * np.abs(
            amptriplets.flatten())**(1/3) + model[2] * amptriplets.flatten()

    return est_closures


def get_adjacent_triplets(num):
    '''
        Return an array of indexes corresponding only to triplets immediately adjacent to each other
    '''
    triplets = []
    for i in range(num - 2):
        triplets.append([i, i+1, i + 2])

    return np.sort(np.array(triplets))


def get_triplets(num, force=None, all=False):
    '''
        Get the indexes of a unique set of combinations of the SLCs
    '''
    numbers = np.arange(0, num, 1)
    permutations = np.array(list(itertools.permutations(numbers, 3)))
    if all:
        return permutations
    combinations = np.sort(permutations, axis=1)
    combinations = np.unique(combinations, axis=0)
    if force is not None:
        combinations = np.array(
            [triplet for triplet in combinations if triplet[0] == force])

    return combinations


def build_A(triplets, coherence):
    '''
      Using the triplet SLC indexes and an array of phi_indexes that
    '''

    phi_indexes = collapse_indexes(coherence)

    A = np.zeros((triplets.shape[0], len(phi_indexes)))
    for i in range(triplets.shape[0]):
        a = A[i]
        triplet = triplets[i]
        a[np.where(phi_indexes == f'{triplet[0]}{triplet[1]}')] = 1
        a[np.where(phi_indexes == f'{triplet[0]}{triplet[2]}')] = -1
        a[np.where(phi_indexes == f'{triplet[1]}{triplet[2]}')] = 1

    rank = np.linalg.matrix_rank(A)
    return A, rank


def get_triplet_covariance(cov, A, n, diagonal=False):
    '''
        Get the covariance between phase closures/triplets
    '''
    phi_cov, indexes = get_phase_covariance(cov, count=n)
    phi_cov = np.swapaxes(phi_cov, 0, 2)
    phi_cov = np.swapaxes(phi_cov, 1, 3)
    triplet_covariance = A @ phi_cov @ A.T

    return triplet_covariance, indexes, A


def get_closure_significance(closures, triplet_covariance, N):
    '''
        Compute a chi^2 test to determine where phase closures cannot be determined by decorrelation alone.
    '''
    k = special.comb(N-1, 2)
    print(f'{k} degrees of freedom')
    closures = closures[:, :, :, np.newaxis]
    print(triplet_covariance.shape)
    S = np.transpose(closures, (0, 1, 3, 2)
                     ) @ np.linalg.inv(triplet_covariance) @ closures

    return 1 - chi2.cdf(S, df=k), S


def write_closures(coherence, folder):
    '''
        Write a collection of closure maps to a given folder
    '''
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

    if len(closures.shape) == 2:
        closures = closures[:, :, np.newaxis]
        trip_covar = trip_covar[:, :, np.newaxis, np.newaxis]

    pval, S = get_closure_significance(closures, trip_covar, N=N)
    pval = np.squeeze(pval).astype(np.float32)

    folder_path = os.path.join(
        '/', folder)
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


def collapse_indexes(cov):
    '''
        Get the canonical indexes of the phases of a given covariance matrix for input into other functions
    '''
    index_matrix = []
    for i in range(cov.shape[0]):
        row = []
        for j in range(cov.shape[1]):
            row.append(f'{i}{j}')
        index_matrix.append(row)

    cov_indexes = np.array(index_matrix)
    cov_indexes = cov_indexes[np.triu_indices(cov_indexes.shape[0], 1)]
    return cov_indexes


def get_phase_covariance(cov, count=0):
    '''
        Compute covariance between the phases of an SLC covariance matrix
    '''
    def get_phi_cov(cov, i, j, k, l):
        '''
        Compute the covariance between two interferograms:
        cov - The first order covariance of the SLCs
        i,j - the indexes of the first interogram
        k,l - the indexes of the second interferogram
        '''
        norm = (1 / (2 * (count/2) *
                     np.abs(cov[i, j]) * np.abs(cov[k, l])))

        c1 = np.angle(cov[i, j] * cov[j, k] * cov[i, k].conj())
        c2 = np.angle(cov[i, j] * cov[j, l] * cov[i, l].conj())
        c3 = np.angle(cov[j, k] * cov[k, l] * cov[j, l].conj())

        t1 = np.abs(cov[i, k]) * np.abs(cov[j, l]) * np.cos(c1 - c3)
        t2 = np.abs(cov[i, l]) * np.abs(cov[j, k]) * np.cos(c2 + c3)
        print('Computed Phase Variance')
        return ((t1 - t2) * norm).astype(np.float64)

    cov_indexes = collapse_indexes(cov)
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


def coherence_to_phivec(coherence: np.complex64) -> np.complex64:
    '''
      Collapse a coherence matrix into a vector of phases
    '''
    utri = np.triu_indices_from(coherence, 1)
    phi_vec = coherence[utri]

    return phi_vec


def phivec_to_coherence(phi_vec: np.complex64, n: np.int8) -> np.complex64:
    '''
        Convert a vector of complex phases back into a coherence matrix
    '''

    coherence = np.ones((n, n), dtype=np.cdouble)
    utri = np.triu_indices_from(coherence, 1)
    coherence[utri] = phi_vec
    coherence = coherence * coherence.T.conj()

    return coherence


def least_norm(A, closures: np.float32 or np.float64, pseudo_inv=None, pinv=False, C=None) -> np.complex64:
    '''
        Solve: Ax = b
        Find the minimum norm vector 'x' of phases that can explain 'b' phase closures

        b should be a float in radians
    '''

    if pseudo_inv is not None:
        return np.exp(1j * pseudo_inv @ closures)

    if C is None:
        C = np.eye(A.shape[1])
    iC = np.linalg.inv(C)

    if pinv and not pseudo_inv:
        phis = np.linalg.pinv(A) @ closures
        # phis = A.T.conj() @ closures
        # return phis / phis.shape[0]
    else:
        phis = A.T @ np.linalg.inv(A @ A.T) @ closures

    return np.exp(1j * phis)


def phi_to_closure(A, phic) -> np.complex64:
    '''
        Use matrix mult to generate vector of phase closures such that 
        the angle xi = phi_12 + phi_23 - phi_13
    '''
    return np.exp(1j * A @ np.angle(phic))
