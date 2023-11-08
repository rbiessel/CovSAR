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
import graphs
from closig import expansion
from closig.plotting import triangle_plot
import colorcet as cc

dezan = SMForward(imag_slope=0.1, r_A=0.01, r_B=0, r_C=4, omega=1.5e9)
dezan = SMForward(imag_slope=0.1, r_A=0.005, r_B=0, r_C=4)
# dezan = SMForward(imag_slope=0.1, r_A=0.0, r_B=0.15, r_C=4)


# dezan = SMForward(imag_slope=0.2, r_A=0.005, r_B=0, r_C=4)
# dezan = SMForward(imag_slope=0.01, r_A=0.005, r_B=0, r_C=4)

# dezan = SMForward(imag_slope=0.05, r_A=0.005, r_B=0, r_C=4)

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


def decay_model(
        R=500, L=100, P=40, coh_decay=0.9, coh_infty=0.1, incoh_bad=None,
        cphases=None, rng=None, returnSigma=False):
    # using the Cao et al. phase convention
    # From s. Zwieback's G reg
    Sigma = ((1 - coh_infty) * toeplitz(np.power(coh_decay, np.arange(P)))
             + coh_infty * np.ones((P, P))).astype(np.complex128)
    if returnSigma:
        return Sigma
    if incoh_bad is not None:
        intensity = Sigma[P // 2, P // 2]
        Sigma[P // 2, :] *= incoh_bad
        Sigma[:, P // 2] *= incoh_bad
        Sigma[P // 2, P // 2] = intensity
    y = circular_normal((R, L, P), Sigma=Sigma, rng=rng, cphases=cphases)
    return y


def read_sm_from_file(path, n=None):
    df = pd.read_csv(path, sep=",", header=1)

    # df.columns = df.columns.str.replace(' ', '')

    # years = df['Yr'].str.strip().values[1:]
    # months = df['Mo'].str.strip().values[1:]
    # days = df['Day'].str.strip().values[1:]
    # hours = df['Hr'].values[1:].astype(str)
    # mins = df['Min'].str.strip().values[1:]

    # dates = (years + '-' + months + '-' + days +
    #          'T' + hours + ':' + mins)
    # dates = dates[np.where(dates != '--')]
    # dates = pd.to_datetime(dates)

    # WASM = df['SM-1'].values[1:].astype(np.float32)
    WASM = df["Water Content, m³/m³ (LGR S/N: 21152731, SEN S/N: 21153846)"].values
    # print(dates)
    # print(WASM)

    if n is not None:

        si = int(np.floor(len(WASM)/n)) + 1

        WASM = WASM[::si]

    return WASM * 100


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

    # velocity, amplitude = get_velocity(base_velocity=1, scaling_factor=3)

    scenario = 'soilMoisture'
    # scenario = 'precipitation'
    scenario = 'velDif'

    coherence_scenario = 'high'
    # coherence_scenario = 'low'

    # plt.imshow(np.abs(C_decay))
    # plt.show()

    # return

    if 'velDif' in scenario:
        velocity, amplitude = get_velocity(
            base_velocity=0.5, scaling_factor=9, filter=40)

        # velocity = np.random.exponential(scale=3, size=velocity.shape) + 1
        shape = velocity.shape

        velocity_noise = np.random.normal(
            loc=1, scale=0.1, size=(shape[0] * shape[1])).reshape(shape)

        velocity += velocity_noise

        sns.kdeplot(velocity.flatten(), bw_adjust=0.5)
        plt.show()
    else:
        velocity, amplitude = get_velocity(
            base_velocity=0, scaling_factor=0, filter=20)
        shape = velocity.shape

    # amplitude = 1 - amplitude
    print(f'Shape: {shape}')

    # velocity m/yr  > cm / day

    velocity = np.abs(velocity)
    velocity = -1 * velocity / 365

    if 'velDif' in scenario:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        vel_im = ax.pcolormesh(velocity * 365)
        ax.set_aspect('equal')
        clb = fig.colorbar(vel_im, ax=ax, label='Velocity')
        ax.set_xlabel('[$\mathrm{m}$]')
        ax.set_ylabel('[$\mathrm{m}$]')
        clb.ax.set_title('[$\mathrm{cm}/\mathrm{year}$]')
        plt.tight_layout()
        plt.savefig(
            '/Users/rbiessel/Documents/igarss_paper/figures/vfield.png', dpi=300, transparent=True)
        plt.show()

    # amplitude = 1
    n = 40  # length of timeseries
    p = 12  # repeat time
    days = np.arange(0, p*n, p)
    baslines = np.tile(days, (len(days), 1))
    baslines = baslines.T - baslines
    if coherence_scenario == 'low':
        C_decay = decay_model(
            coh_decay=0.4, coh_infty=0.01, returnSigma=True, P=n)
    elif coherence_scenario == 'high':
        C_decay = decay_model(coh_decay=1, coh_infty=0.9,
                              returnSigma=True, P=n)

    # np.random.seed(10)

    sm_stack = np.zeros((n, amplitude.shape[0], amplitude.shape[1]))
    sm = np.random.normal(loc=30, scale=5, size=n)
    # Define the "soil moisture" trend for the vegetation
    #

    if 'soilMoisture' in scenario:
        sm = np.linspace(10, 130, n) + np.random.normal(scale=0, loc=0, size=n)
        dezan.set_moistures(sm)
        dezan.plot_dielectric()
        # sm = np.random.normal(scale=5, loc=30, size=n)

    # return

    smdif = np.tile(sm, (len(sm), 1))
    smdif = smdif.T - smdif

    def get_interferogram_dezan(i1, i2, sms, plot=False):
        intf = dezan.get_phases_dezan(sms[i2], sms[i1])
        if plot:
            plt.imshow(np.angle(intf), cmap=plt.cm.seismic,
                       vmin=-np.pi, vmax=np.pi)
            plt.show()
        return intf

    for i in range(n):
        base_sm = sm[i]
        sm_stack[i] += base_sm  # * amplitude

        # sm_stack[i] += np.abs(np.ones(sm_stack[i].shape) * np.random.normal(
        #     loc=70, scale=5)) * (1-amplitude)
        sm[i] = np.mean(sm_stack[i])

        # plt.imshow(sm_stack[i], vmin=0, vmax=50)
        # plt.show()

    plt.plot(days, sm)
    plt.xlabel('Time [$days$]')
    plt.ylabel('Soil Moisture [$frac{m^3}{m^3}$]')
    plt.show()

    # dezan.set_moistures(sm)
    # dezan.plot_dielectric()

    coherence = np.ones((n, n), dtype=np.complex64)
    true_coherence = np.ones((n, n), dtype=np.complex64)

    print('Computing Observed Phase')

    # Look over time indexes
    for i in range(n):
        for j in range(i, n):
            phi = np.exp(
                1j * ((velocity * (days[j] - days[i])) * 4 * np.pi / 5.5465))

            phi_sm = get_interferogram_dezan(i, j, sm_stack)

            # coherence[i, j] = np.mean(
            #     phi.real + phi_sm.real) + 1j * np.mean(phi.imag + phi_sm.imag)

            if 'soilMoisture' in scenario:
                coherence[j, i] = np.mean(phi_sm)

            if 'velDif' in scenario:
                coherence[j, i] = np.mean(
                    phi.real) + 1j * np.mean(phi.imag)

            # MEAN
            true_coherence[j, i] = np.exp(
                1j * (np.mean((velocity) * (days[j] - days[i])) * 4 * np.pi / 5.5465))

            coherence[i, j] = coherence[j, i].conj()
            true_coherence[i, j] = true_coherence[j, i].conj()

    # coherence_mod = np.ones((n, n), dtype=np.complex64)
    # coherence_mod[smdif > 0] = 0.5
    # coherence *= C_decay

    plt.imshow(np.angle(coherence), cmap=plt.cm.seismic)
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    cmap, vabs = cc.cm['CET_D1A'], 90

    twoHops = expansion.TwoHopBasis(coherence.shape[0])
    evalC = twoHops.evaluate_covariance(coherence, compl=True)
    triangle_plot(
        twoHops, evalC, ax=ax[0], cmap=cmap, vabs=vabs)

    smallSteps = expansion.SmallStepBasis(coherence.shape[0])
    evalC = smallSteps.evaluate_covariance(coherence, compl=True)

    triangle_plot(
        smallSteps, evalC, ax=ax[1], cmap=cmap, vabs=vabs)

    plt.show()

    # errors = build_nl_coherence_dezan(sm, dezan)
    # diag_errors = np.diag(errors, k=1)
    # error_ts_bw1 = np.cumprod(diag_errors)
    # coherence = coherence * errors

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
    # plt.scatter(baslines.flatten(), np.angle(
    #     error_pred_evd.flatten()), alpha=0.5, label='EVD (Low Coherence)')
    plt.scatter((baslines).flatten(), np.angle(
        error_coh.flatten()) * (np.angle(
            coherence * true_coherence.conj())).flatten().conj(), s=10)
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
        ind = 10

        error[m] = 365 * error_m[ind] / (ind * 12)

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

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set(ylim=(-1.5, 1.5))
    scat = ax.scatter(baslines.flatten(),
                      error_full[0, :].flatten(), alpha=0.5, label='EVD Predicted Error', s=5, color='black')

    ax.scatter(baslines.flatten(), np.angle(
        coherence * true_coherence.conj()).flatten(), label='True Error', s=5, color='slategray', marker='x')
    ax.text(
        1.1, 0.7, r'EVD Predicted Error $\mathbf{= arg(\hat C \circ \hat \theta \hat \theta^T)}$'
        '\n'
        r'$\mathbf{\hat C}$: Covariance Matrix'
        '\n'
        r'$\mathbf{\hat \theta}$: Estimated Phase History', fontsize=10, transform=ax.transAxes, weight="bold", color='black')

    ax.set_ylabel('Predicted Phase Error [$rad$]')
    ax.set_xlabel('Temporal Baseline [$days$]')
    ax.legend(bbox_to_anchor=(1.1, 1))

    def animate(i):
        y_i = error_full[i, :].flatten()
        scat.set_offsets(np.c_[baslines.flatten(), y_i])
        ax.set_title(f'Max Basline: {i * 12} days', fontsize=12)

    anim = FuncAnimation(
        fig, animate, interval=100, frames=len(subsets)-1)

    plt.subplots_adjust(right=0.6)
    plt.draw()
    plt.show()
    anim.save(
        f'/Users/rbiessel/Documents/igarss_paper/figures/predictedError_{scenario}_{coherence_scenario}.gif')

    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/error_{scenario}_{coherence_scenario}', error)
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/rank_{scenario}', rank)
    np.save(
        f'/Users/rbiessel/Documents/igarss_paper/data/maxbaseline', subsets)

    # cycle rank vs error

    N = 2
    ranks = np.zeros((N))
    error = np.zeros((N))

    for m in range(N):
        k = np.random.uniform(high=1, low=0.5, size=1)
        l = np.random.uniform(high=n-2, low=1, size=1)
        # l = 0
        # k = 1
        G, A = graphs.get_rand_graph(n, k=k, l=l)
        ranks[m] = graphs.cycle_rank(G)

        A = A[:, :, np.newaxis, np.newaxis]
        coherence_reduced = expanded_coh * A

        # plt.imshow(A[:, :, 0, 0])
        # plt.show()
        # plt.imshow(np.abs(coherence_reduced)[:, :, 0, 0])
        # plt.show()

        ph = evd.eig_decomp(coherence_reduced)[0, 0]
        error_m = ph * ph_true.conj()
        error_m = np.angle(error_m) * 56 / (np.pi * 4)

        error[m] = np.sqrt(np.linalg.norm(error_m, 2) / n)
        # error[m] = 365 * error_m[15] / (15 * 12)

    plt.scatter(ranks, error, s=10, color='black', alpha=0.7)
    plt.xlabel('Cycle Rank')
    plt.ylabel('Error')
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].plot(subsets, cond)
    # ax[0].set_xlabel('Max Temporal Baseline')
    # ax[0].set_ylabel('Condition Number')
    # ax[1].plot(subsets, rank)
    # ax[1].set_xlabel('Max Temporal Baseline')
    # ax[1].set_ylabel('Rank')
    # ax[2].plot(subsets, error)
    # ax[2].set_xlabel('Max Temporal Baseline')
    # ax[2].set_ylabel('L2 Error')

    # plt.show()

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

    coherence_5 = sarlab.reduce_cov(expanded_coh, keep_diag=5)
    coherence_8 = sarlab.reduce_cov(expanded_coh, keep_diag=8)

    coherence_2 = sarlab.reduce_cov(expanded_coh, keep_diag=2)

    phase_history_full = evd.eig_decomp(expanded_coh)[0, 0]
    # phase_history_full *= phase_history_full.conj()

    phase_history_2 = evd.eig_decomp(
        coherence_2)[0, 0] * phase_history_full.conj()
    phase_history_5 = evd.eig_decomp(
        coherence_5)[0, 0] * phase_history_full.conj()
    phase_history_8 = evd.eig_decomp(
        coherence_8)[0, 0] * phase_history_full.conj()

    # plt.plot(np.angle(ph_true) * 56 / (np.pi * 4), label='TRUE')
    plt.plot(np.angle(phase_history_8) * 56 / (np.pi * 4), label='max(tb) = 8')
    plt.plot(np.angle(phase_history_5) * 56 / (np.pi * 4), label='max(tb) = 5')
    plt.plot(np.angle(phase_history_2) * 56 / (np.pi * 4), label='max(tb) = 2')
    plt.plot(np.angle(phase_history_full) * 56 / (np.pi * 4), label='full')
    plt.legend(loc='best')
    plt.xlabel('Time [$days$]')
    plt.ylabel('Error [$mm$]')
    plt.show()


main()
