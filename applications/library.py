import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter, median_filter
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
from lc_filter import filter as lc_filter
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


def intensity_to_epsilon(intensities, scaling=0.046):
    print(intensities.shape)
    intensities /= scaling
    intensities = intensities - intensities[:, :, 0, np.newaxis]

    min = np.min(intensities, axis=2)
    print(min.shape)
    intensities += np.abs(min)[:, :, np.newaxis]
    return intensities.astype(np.complex64) + 0j


def compute_tc(cov, phi_est):
    kappa = np.zeros(cov[0, 0].shape, dtype=np.complex64)
    N = phi_est.shape[2]
    for i in range(N):
        for j in range(N):
            if i != j:
                kappa += np.exp(1j * np.angle((cov[i, j] *
                                               (phi_est[:, :, j] * phi_est[:, :, i].conj()).conj())))
    return kappa / (N**2 - N)


def get_intensity(cov):
    return np.log10(np.abs(np.diagonal(cov)))


def gen_lstq(x, y, W=None, C=None, function='linear'):
    if function is 'linear':
        G = np.stack([x]).T
    if function is 'lineari':
        G = np.stack([np.ones(x.shape[0]), x]).T
    elif function is 'root3':
        G = np.stack([x, np.sign(x) * np.abs(x)**(1/3)]).T
    elif function is 'root5':
        G = np.stack([x, np.sign(x) * np.abs(x)**(1/3),
                      np.sign(x) * np.abs(x)**(1/5)]).T

    print(f'Cond(G): {np.linalg.cond(G)}')
    if C is not None:
        C_inv = C
    else:
        C_inv = np.eye(G.shape[0])

    if W is not None:
        Gw = W @ G
        dw = W @ y
        covm = np.linalg.inv(Gw.T @ Gw)
        m = covm @ Gw.T @ dw
    else:
        m = np.linalg.inv(G.T @ C_inv @ G) @ G.T @ C_inv @ y
        covm = None

    return np.flip(m), covm


def fit_cubic(x, y):
    stack = np.stack([np.ones(x.shape[0]), x, np.sign(x) * np.abs(x)**(1/3)]).T
    fit = np.linalg.lstsq(stack, y)
    return fit[0]


def fit_phase_ratio(iratio, nlphase, degree):
    '''
        Fit the non-linear phases to a function of the intensity ratio
    '''
    G = np.zeros((degree - 2, iratio.shape[0])).T
    for i in range(degree - 2):
        G[:, i] = np.array(iratio)**(i+2)

    return np.flip(np.hstack(([0, 0], np.linalg.lstsq(G, nlphase)[0])))


def intensity_closure(i1, i2, i3):
    triplet = ((i2 - i1) * (i3 - i2) * (i1 - i3))
    triplet = triplet * (i1 * i2 * i3)
    return triplet
    # return ((i2 - i1) + (i3 - i2) + (i1 - i3))
    return np.sign(triplet) * (np.abs(triplet))**(1/3)


def phase_closure(phi12, phi23, phi13):
    return phi12 * phi23 * phi13.conj()


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


def interfere(stack, refi, seci, scaling=None, sample=None, show=True, ml=(1, 1), landcover=None, aspect=7, cov=False, sig=None):
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

    if sample is not None:
        interferogram = interferogram[::sample[0], ::sample[1]]

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


def landcover_filter(image, landcover, size, get_count=False, stat='mean', real_only=False):
    '''
        Use a landcover based guassian filter to multilook a complex image.
    '''
    filtered_real, count = lc_filter(
        image.real, landcover, size, stat=stat, real=True)

    if get_count:
        return count

    if real_only:
        return filtered_real

    filtered_imag, count = lc_filter(
        image.imag, landcover, size, stat=stat, real=False)

    imf = filtered_real + 1j * filtered_imag

    return imf


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
