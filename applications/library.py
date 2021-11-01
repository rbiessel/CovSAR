import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage
from scipy import stats, special
from scipy.stats import chi2
from scipy import linalg
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.decomposition import PCA, FastICA
from mpl_toolkits.mplot3d import Axes3D
import os
import psutil


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


def compute_tc(cov, phi_est):
    print('Computing TC')
    kappa = np.zeros(cov[0, 0].shape, dtype=np.complex64)
    N = phi_est.shape[2]
    for i in range(N):
        for j in range(N):
            if i != j:
                kappa += np.exp(1j * np.angle((cov[i, j] *
                                               (phi_est[:, :, j] * phi_est[:, :, i].conj()).conj())))
    return kappa / (N**2 - N)


def get_intensity(cov):
    return np.log(np.abs(np.diagonal(cov)))


def get_closure_significance(triplet, triplet_variance):
    k = special.comb(2, 2)
    print(f'{k} degrees of freedom')

    S = np.angle(triplet) * triplet_variance**-1 * np.angle(triplet)

    return 1 - chi2.cdf(S, df=k), S


def non_local_filter(image, sig):
    print('Performing Non Local Means denoising on an input image... ')
    denoise_fast = denoise_nl_means(
        image, h=0.8 * sig, sigma=sig, fast_mode=True)
    return denoise_fast


def non_local_complex(image, sig):
    print('Performing Non Local Means denoising on an input image... ')
    return denoise_nl_means(
        image.real, h=0.8 * sig, sigma=sig, fast_mode=True) + 1j * denoise_nl_means(
            image.imag, h=0.8 * sig, sigma=sig, fast_mode=True)


def multilook(im, ml=(8, 2), thin=(8, 2)):
    '''
        Use a uniform guassian filter to multilook a complex image.
    '''
    outshape = (im.shape[0] // ml[0], im.shape[1] // ml[1])
    imf = uniform_filter(im.real, size=ml) + 1j * \
        uniform_filter(im.imag, size=ml)
    # imf = imf[::ml[0]//2, ::ml[1]//2].copy()[:outshape[0], :outshape[1]]
    return imf


def remove_sm_phase_pcr(interferogram, delta_sm, closures=None):

    delta_sm = delta_sm.real
    if closures is None:
        closures = np.zeros(interferogram.shape) + 0.01

    closure_thresh = 0
    closure_abs = np.abs(np.angle(closures))
    og_shape = interferogram.shape

    sm_mean = np.mean(delta_sm)
    sm_std = np.std(delta_sm)

    delta_sm = (delta_sm - sm_mean) / sm_std

    int_mean = np.mean(interferogram)
    int_std = np.std(interferogram)

    interferogram = (interferogram - int_mean) / int_std

    new_sm = delta_sm[closure_abs >= closure_thresh]
    new_int = interferogram[closure_abs >= closure_thresh]

    samples = np.array(
        [new_sm.flatten(), new_int.real.flatten(), new_int.imag.flatten()]).T

    data_all = np.array(
        [delta_sm.flatten(), interferogram.real.flatten(), interferogram.imag.flatten()]).T

    # data_all = samples
    # print(samples.shape)
    # samples = samples[np.random.choice(samples.shape[0], 6000, replace=True)]

    pca = PCA(n_components=3)
    pca.fit(samples)
    transformed = pca.transform(data_all)
    # transformed = transformed / np.std(transformed)

    rng = np.random.RandomState(32)
    ica = FastICA(random_state=rng).fit(samples)
    # print(ica.mixing_)
    # print('ICA FIT: ', ica.mixing_)
    S_ica_ = ica.transform(data_all)

    S_ica_ = S_ica_ / np.std(S_ica_)

    V = pca.components_.T
    V = ica.mixing_.T

    # print('PCA vectors: ', V)
    V = V / np.linalg.norm(V)
    # print(transformed)
    y = np.dot(data_all, V[0])
    y2 = np.dot(data_all, V[1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # axes[0].scatter(data_all[:, 0],
    #                 np.angle((data_all[:, 1] + 1j * data_all[:, 2])).flatten(), alpha=.3, label='samples', s=2)
    # axes[0].set(xlabel='Original Samples', ylabel='y')
    # axes[1].scatter(S_ica_[:, 0], np.angle(
    #     (S_ica_[:, 1] + 1j * S_ica_[:, 2])).flatten(), alpha=.3, s=2)
    # axes[1].set(xlabel='corrected', ylabel='y')

    # plt.tight_layout()
    # plt.show()

    corrected = S_ica_[:, 1] + (S_ica_[:, 2] * 1j)
    corrected = (corrected * int_std) + int_mean

    # print(f'Og Shape: {og_shape}, Corrected Shape: ', corrected.shape)

    # interferogram = (np.ones(og_shape) + (np.zeros(og_shape) * 1j)).flatten()
    # closures = closures.flatten()

    # interferogram[closures > closure_thresh] = corrected

    # interferogram = np.reshape(interferogram, og_shape)

    return np.reshape(corrected, og_shape)


def remove_sm_phase_pca(interferogram, delta_sm):

    # Convert to  columns: [sm, interferogram]
    og_shape = interferogram.shape

    delta_sm = (delta_sm - np.mean(delta_sm)) / np.std(delta_sm)

    int_mean = np.mean(interferogram)
    int_std = np.std(interferogram)

    interferogram = (interferogram - int_mean) / int_std

    samples = np.array(
        [delta_sm.flatten(), interferogram.flatten()]).T

    data = samples
    print(samples.shape)
    # samples = samples[np.random.choice(samples.shape[0], 6000, replace=True)]

    pca = PCA(n_components=2)
    pca.fit(samples)
    V = pca.components_.T

    print(V)

    y = np.dot(data, V[0])
    y2 = np.dot(data, V[1])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].scatter(data[:, 0],
                    data[:, 1], alpha=.3, label='samples', s=2)
    axes[0].set(xlabel='Original Samples', ylabel='y')
    axes[1].scatter(data[:, 0], y, alpha=.3, s=2)
    axes[1].set(xlabel='Projected data onto first PCA component', ylabel='y')
    axes[2].scatter(data[:, 0], y2, alpha=.3, s=2)
    axes[2].set(
        xlabel='Projected data onto second PCA component', ylabel='y')
    plt.tight_layout()
    plt.show()

    return (np.dot(data, V[1]).reshape(og_shape) * int_std) + int_mean


def get_db_matrix(stack, sig=1, size=None, landcover=None):
    print('current stack: ', stack.shape)

    slcn = stack.shape[0]
    db_dif = np.zeros(
        (slcn, slcn, stack.shape[1], stack.shape[2]), dtype=np.float64)

    stack = np.abs(stack)**2
    # stack[stack < 0.0001] = 0.0001

    # filter here

    # ts = []
    # for image in stack:
    #     ts.append(non_local_filter(image, sig=sig)[100, 100])

    for i in range(slcn):
        for j in range(slcn):
            if j >= i:
                if size is None:
                    db_dif[i, j, :, :] = non_local_filter(
                        stack[i, :, :], sig=sig) / non_local_filter(stack[j, :, :], sig=sig)

                else:
                    if landcover is None:
                        db_dif[i, j, :, :] = uniform_filter(stack[i, :, :], size=(
                            size, size)) / uniform_filter(stack[j, :, :], size=(size, size))
                        # plt.imshow(db_dif[i, j])
                        # plt.show()

                    else:
                        db_dif[i, j, :, :] = landcover_filter(
                            stack[i, :, :], size=size, landcover=landcover, real_only=True) / landcover_filter(stack[j, :, :], size=size, landcover=landcover, real_only=True)
            else:
                db_dif[i, j, :, :] = 1 / db_dif[j, i, :, :]

    db_dif = np.log(db_dif)

    return db_dif


def interfere(stack, refi, seci, scaling=None, show=True, ml=(1, 1), landcover=None, aspect=7, cov=False, sig=None):
    '''
        Generate a multilooked interferogram from two SLC images of the same dimension
    '''

    if refi == seci and not cov:
        return stack[refi] * stack[seci].conj()
    else:
        if scaling is None:
            scale1 = np.abs(interfere(stack, refi, refi, show=False,
                                      ml=ml, landcover=landcover))
            scale2 = np.abs(interfere(stack, seci, seci, show=False,
                                      ml=ml, landcover=landcover))
            scaling = np.sqrt(scale1 * scale2)

    interferogram = stack[refi] * stack[seci].conj()
    if scaling is not 1:
        scaling[scaling == 0] = 1
        interferogram = interferogram / scaling
    # interferogram[np.isnan(interferogram)] = 0
    if ml[0] > 1 and ml[1] > 1:
        if landcover is not None:
            interferogram = landcover_filter(
                interferogram, landcover, ml[0], stat='mean')
        elif sig is not None:
            interferogram = non_local_complex(interferogram, sig=sig)
        else:
            interferogram = multilook(interferogram, ml=(ml[0], ml[1]))

    if show:
        fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax1[0].imshow(np.abs(interferogram),
                      cmap='binary_r', origin='lower', vmin=0, vmax=1)
        ax1[0].title.set_text('Coherence')

        ax1[1].imshow(np.angle(interferogram),
                      cmap=plt.cm.hsv, origin='lower', vmin=-np.pi, vmax=np.pi)
        ax1[1].title.set_text('Phi')

        if aspect is not None:
            ax1[0].set_aspect(aspect)
            ax1[1].set_aspect(aspect)
        plt.show()

    return interferogram


def cov2coh(cov):
    coh = np.zeros(cov.shape)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            print(i, j)
            scal1 = np.abs(cov[i, i])
            scal2 = np.abs(cov[j, j])
            coh[i, j, :, :] = cov[i, j, :, :] / np.sqrt(scal1 * scal2)
            plt.imshow(np.abs(coh[i, j]))
            plt.show()
    return coh


def get_covariance(stack, ml='landcover', ml_size=(20, 4), landcover=None, coherence=False, sig=None):
    print('current stack: ', stack.shape)
    if landcover is not None:
        print('Landcover size: ', landcover.shape)
    slcn = stack.shape[0]
    cov = np.zeros(
        (slcn, slcn, stack.shape[1], stack.shape[2]), dtype=np.complex64)

    scaling = 1
    if coherence:
        scaling = None

    for i in range(slcn):
        for j in range(slcn):
            if j >= i:
                cov[i, j, :, :] = interfere(
                    stack, i, j, ml=ml_size, show=False, aspect=1, scaling=scaling, cov=True, landcover=landcover, sig=sig)
            else:
                cov[i, j, :, :] = cov[j, i, :, :].conj()

    return cov


def landcover_filter(image, landcover, size, get_count=False, stat='mean', real_only=False):
    '''
        Use a landcover based guassian filter to multilook a complex image.
    '''

    print('Performing a landcover driven filter')

    filtered_real, count = lc_filter(image.real, landcover, size, stat=stat)

    if get_count:
        return count

    if real_only:
        return filtered_real

    filtered_imag, count = lc_filter(image.imag, landcover, size, stat=stat)

    imf = filtered_real + 1j * filtered_imag

    return imf


def calcClosure(stack, one, two, three, ml=(1, 1), show=True, landcover=None, aspect=7, sig=None):
    '''
        Generate phase closures from a stack of SLC images and three indexes
    '''

    phi12 = interfere(stack, one, two, show=False,
                      ml=ml, landcover=landcover, sig=sig)
    phi23 = interfere(stack, two, three, show=False,
                      ml=ml, landcover=landcover, sig=sig)
    phi13 = interfere(stack, one, three, show=False,
                      ml=ml, landcover=landcover, sig=sig)
    closure = phi12 * phi23 * phi13.conj()

    if show:
        fig, ax = plt.subplots()
        ax.imshow(np.angle(closure), cmap=plt.cm.RdBu_r,
                  vmin=-np.pi, vmax=np.pi)

        if aspect is not None:
            ax.set_aspect(aspect)

        plt.title('Phase Closure')
        plt.show()

        # plt.imshow(10 * np.log(np.abs(closure)), cmap=plt.cm.gray)
        # plt.show()

    return closure


def clone_isce(data, outfile):
    im2 = createImage()
    im2.setWidth(data.shape[1])
    im2.setLength(data.shape[0])
    im2.setAccessMode('write')
    im2.filename = outfile
    im2.dataType = 'FLOAT'
    im2.createImage()

    im2.dump(f'{outfile}.xml')
    data.tofile(outfile)
