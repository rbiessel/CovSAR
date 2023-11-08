from covariance import CovarianceMatrix
from cv2 import mean
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
import closures
import figStyle
from sm_forward.sm_forward import SMForward
import library as sarlab
from scipy.stats import gaussian_kde
import pandas as pd
from pl import evd
import seaborn as sns
from scipy.linalg import cholesky, toeplitz
from matplotlib.animation import FuncAnimation
import os
from greg import simulation as greg_sim
from triplets import eval_triplets
from scipy import stats
import matplotlib.patches as mpl_patches
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
dezan = SMForward(imag_slope=0.1, r_A=0.01, r_B=0, r_C=4, omega=1.5e9)
dezan = SMForward(imag_slope=0.05, r_A=0.04, r_B=0, r_C=4)
dezan = SMForward(imag_slope=0.01, r_A=0.008, r_B=0, r_C=4)

sm_path = '/Users/rbiessel/Documents/dalton_SWC/SlopeMountain/SM_SWC1_2022.csv'


def circular_white_normal(size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    y = rng.standard_normal(size=size) + 1j * rng.standard_normal(size=size)
    return y / np.sqrt(2)


# simulate Gaussian speckle with general Sigma, FROM GREG
def circular_normal(size, Sigma=None, rng=None, cphases=None):
    y = circular_white_normal(size, rng=rng)
    if Sigma is not None:
        assert Sigma.shape[0] == Sigma.shape[1]
        assert len(Sigma.shape) == 2
        assert size[-1] == Sigma.shape[0]
        L = cholesky(Sigma, lower=True)
        y = np.einsum('ij,...j->...i', L, y, optimize=True)
        if cphases is not None:
            y = np.einsum('j, ...j->...j', cphases, y, optimize=True)
    return y


def build_nl_coherence(nl_func, sm):
    '''
      Given a time-series of sm, compute a simulated coherence matrix with systematic phase closures
    '''
    coherence = np.zeros((sm.shape[0], sm.shape[0]), dtype=np.cdouble)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = nl_func(sm[i] - sm[j])  # * nl_func(sm[j]).conj()

    return coherence


def sm_to_phase_sqrt(sm):
    '''
      Convert an array of volumetric soil moistures to angles according to a square root function.
      Soil moistures betwen 0 and 50 will return angles in the range [0, pi/2] in the form of a complex array
    '''
    sign = np.sign(sm)
    angle = (np.sqrt(2) * np.pi)**-1 * np.sqrt(np.abs(sm))
    return np.exp(1j * sign * angle)


def build_nl_coherence_dezan(sm, dezan):
    coherence = np.ones((sm.shape[0], sm.shape[0]), dtype=np.complex64)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = dezan.get_phases_dezan(sm[i], sm[j])

    return coherence


def multilook(image, looks):
    image = uniform_filter(image.real, size=looks) + \
        1j * uniform_filter(image.imag, size=looks)

    return image


def get_velocity(scaling_factor=5, base_velocity=1, filter=100):
    '''
        Scaling factor defines how many times faster the

    '''
    # Load root image
    path = './polygon_drawing.jpg'
    col = Image.open(path)
    gray = col.convert('L')
    gray = np.array(gray)
    filtered = uniform_filter(gray, size=filter)[::30, ::30]  # [10:20, 10:20]
    amplitude = filtered / 255
    velocity = 1 - amplitude
    # velocity = amplitude

    velocity += base_velocity
    velocity *= scaling_factor

    return velocity, amplitude


def main():

    n = 20
    shape = (30, 30)

    s = np.random.gamma(1, 1, size=shape)
    # s = np.random.gamma(1, 1, size=shape)
    # s = np.random.uniform(0, 0.1, size=shape)
    s_bar = 0

    # s_bar = 0.5 * 0.1**2

    # sns.kdeplot(s.flatten())
    # plt.show()

    sm = np.random.normal(loc=30, scale=2, size=(n))
    sm[5] = 1
    sm[6] = 35
    # plt.plot(sm)
    # plt.xlabel('time')
    # plt.ylabel('SM [m3/m3]')
    # plt.show()

    C = np.zeros((n, n), dtype=np.complex64)
    true_coherence = np.zeros((n, n), dtype=np.complex64)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            dSm = (sm[i] - sm[j])
            phi = np.exp(1j * s * dSm)
            # plt.imshow(np.angle(phi), cmap=plt.cm.seismic_r, vmin=-np.pi, vmax=np.pi)
            # plt.show()
            C[i, j] = np.mean(phi.real) + 1j * np.mean(phi.imag)

    # plt.imshow(np.angle(C), cmap=plt.cm.seismic_r, vmin=-np.pi, vmax=np.pi)
    # plt.show()

    dSm = np.tile(sm, (len(sm), 1))
    dSm = (dSm.T - dSm)

    plt.scatter(dSm.flatten(), np.angle(C).flatten(), label='Simulated')
    plt.scatter(dSm.flatten(), np.arctan(
        dSm).flatten(), label='Arctan', alpha=0.5)
    plt.xlabel('SM')
    plt.ylabel('Phase')
    plt.legend(loc='best')
    plt.show()

    triplets = closures.get_triplets(C.shape[0], all=False)
    A, rank = closures.build_A(triplets, C)

    intensities = sm / 1.5

    closure_stack = np.zeros(len(triplets), dtype=np.complex64)
    intensity_triplets = np.zeros((len(triplets)), dtype=np.float32)

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = C[triplet[0], triplet[1]] * C[triplet[1],
                                                triplet[2]] * C[triplet[0], triplet[2]].conj()
        print(np.angle(closure))
        amp_triplet = sarlab.intensity_closure(
            intensities[triplet[0]], intensities[triplet[1]], intensities[triplet[2]], norm=False, cubic=False, filter=1, kappa=1, function='arctan')
        # print(amp_triplet)
        closure_stack[i] = closure
        intensity_triplets[i] = amp_triplet

    fig, ax = plt.subplots()
    r, pval = stats.pearsonr(intensity_triplets, np.angle(closure_stack))

    ax.scatter(intensity_triplets, np.angle(
        closure_stack), s=10, color='black', alpha=0.3)

    ax.set_xlabel(r'$\mathfrak{S} [$-$]  $')
    ax.set_ylabel(r'$\Xi$ [$rad$]')
    ax.axhline(y=0, color='k', alpha=0.15)
    ax.axvline(x=0, color='k', alpha=0.15)
    ax.grid(alpha=0.2)

    labels = []
    labels.append(f'R$^{{2}} = {{{np.round(r**2, 2)}}}$')

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                     lw=0, alpha=0)] * 2
    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad

    ax.legend(handles, labels, loc='best', fontsize='medium',
              fancybox=True, framealpha=0.7,
              handlelength=0, handletextpad=0)
    plt.show()
    # return
    coeff, covm = sarlab.gen_lstq(
        intensity_triplets, np.angle(closure_stack), W=None, function='linear')

    systematic_phi_errors = closures.least_norm(
        A, np.angle(closure_stack), pinv=False, pseudo_inv=np.linalg.pinv(A))

    error_coh = closures.phivec_to_coherence(systematic_phi_errors, n)
    plt.scatter(dSm.flatten(), np.angle(error_coh).flatten(), label='Pred')
    plt.scatter(dSm.flatten(), np.angle(C), label='True')
    plt.scatter(dSm.flatten(), np.angle(
        error_coh).flatten() + dSm.flatten() * s_bar, label='test')
    plt.legend(loc='best')
    plt.show()

    plt.scatter(dSm.flatten(), np.angle(
        error_coh).flatten() * np.angle(C).conj().flatten())

    plt.show()

    return
    C = C[:, :, np.newaxis, np.newaxis]
    true_coherence = true_coherence[:, :, np.newaxis, np.newaxis]

    ph = evd.eig_decomp(C[:, :])[0, 0]
    ph = ph[:, np.newaxis]
    ph_coh = ph @ ph.T.conj()

    error_pred_evd = C[:, :, 0, 0] * ph_coh.conj()

    print(C.shape)
    # return
    x = np.linspace(intensity_triplets.min(),
                    intensity_triplets.max(), 100)

    sm_dif = (sm[np.newaxis].T - sm[np.newaxis])

    plt.scatter(sm_dif.flatten(), np.angle(
        error_coh.flatten()), alpha=0.5, label='Least Norm')
    plt.scatter(sm_dif.flatten(), np.angle(
        C[:, :, 0, 0] * true_coherence[:, :, 0, 0].conj()).flatten(), alpha=0.5, label='True Error')
    # plt.scatter(sm_dif.flatten(), np.angle(
    #     error_pred_evd.flatten()), alpha=0.5, label='EVD (Low Coherence)')
    plt.scatter(sm_dif.flatten(), np.angle(
        C[:, :, 0, 0] * true_coherence[:, :, 0, 0].conj() * error_coh[:, :, 0, 0].conj()).flatten(), alpha=0.5, label='Residual')
    plt.legend(loc='best')
    plt.ylabel('Error [$rad$]')
    plt.xlabel('Temporal Baseline [$days$]')
    print(np.mean(np.angle(error_coh)[sm_dif > 0]))
    plt.show()

    # plt.legend(loc='best')
    # plt.ylabel('Error [$rad$]')
    # plt.xlabel('Temporal Baseline [$days$]')
    # plt.show()
    return
    # plt.imshow(np.angle(errors))
    # plt.show()
    # plt.plot(np.angle(diag_errors))
    # plt.plot(sm / np.max(sm))
    # plt.show()

    # print(triplets)
    # # triplets = np.array([triplet for triplet in triplets if triplet[0] == 0])

    # triplets_permuted_1 = [[triplet[0], triplet[2], triplet[1]]
    #                        for triplet in triplets]

    # triplets = np.concatenate(
    #     (triplets, triplets_permuted_1))

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].set_title('Observed Phases')
    # im = ax[0].imshow(np.angle(coherence), vmin=-np.pi/10,
    #                   vmax=np.pi/10, cmap=plt.cm.seismic)
    # ax[1].set_title('True Phases')
    # ax[1].imshow(np.angle(true_coherence), vmin=-np.pi/10,
    #              vmax=np.pi/10, cmap=plt.cm.seismic)
    # ax[2].set_title('Difference')
    # ax[2].imshow(np.angle(coherence * true_coherence.conj()), vmin=-np.pi/10,
    #              vmax=np.pi/10, cmap=plt.cm.seismic)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax,  label='Radians')
    # plt.show()

    plt.scatter((baslines).flatten(), np.angle(
        coherence * true_coherence.conj()).flatten(), s=10)
    plt.xlabel('Temporal Baseline')
    plt.ylabel('Phase Error')
    plt.show()

    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/baselines', (baslines).flatten())

    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/phierrors_{scenario}', np.angle(
            coherence * true_coherence.conj()).flatten())
    kappas = np.linspace(0.1, 0.1, 1)

    triplets = closures.get_triplets(n, all=False)

    A, rank = closures.build_A(triplets, coherence)
    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[: rank].T @ np.diag(1/S[: rank]) @ U.T[: rank]

    sim_closures = np.zeros(len(triplets), dtype=np.complex64)
    amp_triplets = np.zeros(len(triplets))

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        amp_triplet = sarlab.intensity_closure(
            sm[triplet[0]], sm[triplet[1]], sm[triplet[2]], norm=False, cubic=False, legacy=False, kappa=1)

        sim_closures[i] = closure
        amp_triplets[i] = amp_triplet

    plt.scatter(amp_triplets, np.angle(sim_closures))
    plt.show()
    form = 'linear'
    coeff, covm = sarlab.gen_lstq(
        amp_triplets, np.angle(sim_closures), W=None, function='linear')

    x = np.linspace(amp_triplets.min(), amp_triplets.max(), 100)

    xy = np.vstack(
        [amp_triplets.flatten(), np.angle(sim_closures).flatten()])
    z = gaussian_kde(xy)(xy)

    systematic_phi_errors = closures.least_norm(
        A, np.angle(sim_closures), pinv=False, pseudo_inv=A_dagger)

    systematic_phi_errors_alt = closures.least_norm(
        A, np.angle(sim_closures), pinv=False, pseudo_inv=A.T/n)

    error_coh = closures.phivec_to_coherence(systematic_phi_errors, n)
    error_coh_alt = closures.phivec_to_coherence(
        systematic_phi_errors_alt, n)

    sm_dif = (sm[np.newaxis].T - sm[np.newaxis])

    ph = evd.eig_decomp(coherence[:, :, np.newaxis, np.newaxis])[0, 0]
    ph = ph[:, np.newaxis]
    ph_coh = ph @ ph.T.conj()

    error_pred_evd = coherence * ph_coh.conj()

    print('Rank of temporally consistent matrix: ',
          np.linalg.matrix_rank(ph_coh))

    # print(sm_dif.shape)
    print(error_coh.shape)
    plt.scatter(baslines.flatten(), np.angle(
        error_coh.flatten()), alpha=0.5, label='Least Norm')
    plt.scatter(baslines.flatten(), np.angle(
        coherence * true_coherence.conj()).flatten(), alpha=0.5, label='True Error')
    plt.scatter(baslines.flatten(), np.angle(
        error_pred_evd.flatten()), alpha=0.5, label='EVD (Low Coherence)')
    plt.legend(loc='best')
    plt.ylabel('Error [$rad$]')
    plt.xlabel('Temporal Baseline [$days$]')
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/lnpred_{scenario}', np.angle(error_coh))
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/evdpred_{scenario}_{coherence_scenario}', np.angle(error_pred_evd))
    print(np.mean(np.angle(error_coh)[sm_dif > 0]))
    plt.title(f'Max baseline')
    plt.show()

    sns.kdeplot(np.angle(sim_closures), bw_adjust=.4, linewidth=1)
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/closures_{scenario}', np.angle(sim_closures))
    plt.xlabel('Closure Phase')
    plt.ylabel('Probability')
    plt.show()

    expanded_coh = coherence[:, :, np.newaxis, np.newaxis]
    expanded_coh_true = true_coherence[:, :, np.newaxis, np.newaxis]
    subsets = np.arange(2, n, 1)
    cond = np.zeros(subsets.shape)
    error = np.zeros(subsets.shape)
    rank = np.zeros(subsets.shape)

    error_full = np.zeros(
        (subsets.shape[0], len(baslines.flatten())))
    print(error_full.shape)

    ph_true = evd.eig_decomp(expanded_coh_true)[0, 0]

    for m in range(subsets.shape[0]):
        subset = subsets[m]
        coherence_reduced = sarlab.reduce_cov(
            expanded_coh, keep_diag=subset)
        # plt.imshow(np.angle(coherence_reduced[:, :, 0, 0]))
        # plt.show()
        cond[m] = np.linalg.cond(coherence_reduced[:, :, 0, 0])
        rank[m] = np.linalg.matrix_rank(coherence_reduced[:, :, 0, 0])

        ph = evd.eig_decomp(coherence_reduced)[0, 0]
        error_m = ph * ph_true.conj()
        error_m = np.angle(error_m) * 56 / (np.pi * 4)

        # error[m] = np.sqrt(np.linalg.norm(error_m, 2) / n)
        error[m] = 365 * error_m[15] / (15 * 12)

        # print('at 15: ', error_m[15])

        ph = ph[:, np.newaxis]
        ph_coh = ph @ ph.T.conj()

        error_pred_evd = coherence * ph_coh.conj()
        error_full[m, :] = np.angle(error_pred_evd.flatten())

        triplets = closures.get_triplets(n, all=False, max=subset)

        sim_closures = np.zeros(len(triplets), dtype=np.complex64)

        for i in range(len(triplets)):
            triplet = triplets[i]
            closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                    triplet[2]] * coherence[triplet[0], triplet[2]].conj()

            sim_closures[i] = closure

        # sns.kdeplot(np.angle(sim_closures))
        # plt.title(f'maxb: {subset}')
        # plt.show()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set(ylim=(-1.5, 1.5))
    scat = ax.scatter(baslines.flatten(),
                      error_full[0, :].flatten(), alpha=0.5, label='EVD Predicted Error', s=5, color='black')

    ax.scatter(baslines.flatten(), np.angle(
        coherence * true_coherence.conj()).flatten(), label='True Error', s=5, color='slategray', marker='x')
    ax.text(
        0.35, 0.8, r'EVD Predicted Error $\mathbf{= arg(\hat C \circ \hat \theta \hat \theta^T)}$'
        '\n'
        r'$\mathbf{\hat C}$: Covariance Matrix'
        '\n'
        r'$\mathbf{\hat \theta}$: Estimated Phase History', fontsize=10, transform=ax.transAxes, weight="bold", color='black')

    ax.set_ylabel('Predicted Phase Error [$rad$]')
    ax.set_xlabel('Temporal Baseline [$days$]')
    ax.legend(loc='best')

    def animate(i):
        y_i = error_full[i, :].flatten()
        scat.set_offsets(np.c_[baslines.flatten(), y_i])
        ax.set_title(f'Max Basline: {i * 12} days')

    anim = FuncAnimation(
        fig, animate, interval=100, frames=len(subsets)-1)

    plt.draw()
    plt.show()
    anim.save(
        f'/Users/rbiessel/Documents/igarss_paper/figures/predictedError_{scenario}_{coherence_scenario}.mp4')

    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/error_{scenario}_{coherence_scenario}', error)
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/rank_{scenario}', rank)
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/maxbaseline', subsets)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].plot(subsets, cond)
    ax[0].set_xlabel('Max Temporal Baseline')
    ax[0].set_ylabel('Condition Number')
    ax[1].plot(subsets, rank)
    ax[1].set_xlabel('Max Temporal Baseline')
    ax[1].set_ylabel('Rank')
    ax[2].plot(subsets, error)
    ax[2].set_xlabel('Max Temporal Baseline')
    ax[2].set_ylabel('L2 Error')

    plt.show()

    # Plot EIgval spectrum

    # eigval, eigvec = np.linalg.eigh(coherence)

    # plt.plot(eigval)
    # plt.title('Spectrum of C')
    # plt.show()

    if False:
        prev = np.ones(n, dtype=np.complex64)
        for l in range(n, -1, -1):

            ph_l = evd.eig_decomp(expanded_coh, eigvector=l)[0, 0]

            ph_l = ph_l * prev
            prev = ph_l

            ph_l = ph_l[:, np.newaxis]
            ph_l_coh = ph_l @ ph_l.T.conj()

            error_pred_evd_l = coherence * ph_l_coh.conj()

            plt.scatter(baslines.flatten(), np.angle(
                error_pred_evd_l.flatten()), alpha=0.5, label=f'EVD:{l}')
            plt.title(f'{l}')
            plt.show()


main()
