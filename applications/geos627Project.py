from sm_forward.sm_forward import SMForward
import closures
import numpy as np
from matplotlib import pyplot as plt
from scipy import special


dezan = SMForward(imag_slope=1/10, r_A=0.05, r_B=0.1, r_C=4)


def sm_to_phase_sqrt(sm):
    '''
      Convert an array of volumetric soil moistures to angles according to a square root function.
      Soil moistures betwen 0 and 50 will return angles in the range [0, pi/2] in the form of a complex array
    '''
    sign = np.sign(sm)
    angle = (np.sqrt(2) * np.pi)**-1 * np.sqrt(np.abs(sm))
    return np.exp(1j * sign * angle)


def build_nl_coherence_dezan(sm, dezan):
    coherence = np.zeros((sm.shape[0], sm.shape[0]), dtype=np.cdouble)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = dezan.get_phases_dezan(sm[i], sm[j])

    return coherence


def build_nl_coherence(nl_func, sm):
    '''
      Given a time-series of sm, compute a simulated coherence matrix with systematic phase closures
    '''
    coherence = np.zeros((sm.shape[0], sm.shape[0]), dtype=np.cdouble)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = nl_func(sm[i] - sm[j])  # * nl_func(sm[j]).conj()

    return coherence


def build_noise_coherence(n, sigma):
    '''
      Given a time-series of sm, compute a simulated coherence matrix with systematic phase closures
    '''
    coherence = np.ones((n, n), dtype=np.cdouble)
    if sigma == 0:
        return coherence
    for i in range(n):
        for j in range(n):
            if j > i:
                coherence[i, j] = 1/sigma * np.exp(
                    1j * np.random.normal(loc=0, scale=sigma, size=1))
                coherence[j, i] = coherence[i, j].conj()
    return coherence


def time_series_to_coherence(ts):
    '''
      Given a phase time-series, compute a simulated coherence matrix
    '''
    coherence = np.zeros((ts.shape[0], ts.shape[0]), dtype=np.cdouble)

    for i in range(ts.shape[0]):
        for j in range(ts.shape[0]):
            coherence[i, j] = ts[i] * ts[j].conj()
    # verify the coherence matrix is positive semidefinite
    # try:
    #   np.linalg.cholesky(coherence)

    # except:
    #   print('The simulated coherence matrix is not semi-positive definite. Something went wrong!')

    return coherence


def sm_to_matrix(sm):
    sm2d = sm[:, np.newaxis]
    sm_stack = np.tile(sm2d, (sm2d.shape[0]))
    sm_dif = (sm_stack.T - sm_stack)

    return sm_dif


def main():
    ns = np.arange(3, 25, 1)
    k = special.comb(ns - 1, 2)
    phik = special.comb(ns, 2)

    # for n in ns:
    #     triplets = closures.get_triplets(n)
    #     deformation = np.exp(1j * np.linspace(0, 2, n))

    #     A, rank = closures.build_A(triplets, n)[0]
    #     print(rank)
    #     print(np.linalg.cond(A))

    # return
    # plt.plot(phik, k, '.', label='Independent Phase Closures')
    # plt.plot(phik, phik, '--', label='1:1')
    # plt.xlabel('Parameters')
    # plt.ylabel('Observations')
    # plt.legend(loc='upper left')
    # # plt.axis('equal')
    # plt.xlim(0, 250)
    # plt.ylim(0, 250)
    # plt.show()

    n = 10
    deformation = np.exp(1j * np.linspace(0, 2, n))

    sm = np.abs(np.linspace(20, 40, n) + (np.random.rand(n) * 1))
    # sm[3] = sm[4]
    # sm[7] = sm[5]
    sm_difs = closures.coherence_to_phivec(sm_to_matrix(sm))

    nl_coh_sqrt = build_nl_coherence(sm_to_phase_sqrt, sm).conj()

    sigma = 0.01  # radians
    noise_coh = build_noise_coherence(n, sigma)

    dezan.set_moistures(sm)
    # dezan.plot_dielectric()

    nl_coh_dezan = build_nl_coherence_dezan(sm, dezan).conj()

    vmin = -np.pi/2
    vmax = -vmin

    disp_coh = time_series_to_coherence(deformation)

    k = special.comb(disp_coh.shape[0] - 1, 2)
    triplets = closures.get_triplets(disp_coh.shape[0])
    triplets = triplets[0:int(k)]
    A = closures.build_A(triplets, disp_coh)[0]
    print(A)

    U, S, Vh = np.linalg.svd(A)
    print(U)
    rank = np.linalg.matrix_rank(A)
    print('Condition number of A:', np.linalg.cond(A))
    null_space = Vh.T[rank:]
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]
    # A_dagger = np.linalg.pinv(A)

    observed_closures_dezan = closures.phi_to_closure(
        A, closures.coherence_to_phivec(nl_coh_dezan * noise_coh))

    observed_closures_sqrt = closures.phi_to_closure(
        A, closures.coherence_to_phivec(nl_coh_sqrt * noise_coh))
    # Invert phases

    inverted_phases_dezn = np.exp(
        1j * A_dagger @ (np.angle(observed_closures_dezan)))
    inverted_phases_sqrt = np.exp(
        1j * A_dagger @ (np.angle(observed_closures_sqrt)))

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(np.angle(nl_coh_dezan), vmin=vmin,
                    vmax=vmax, cmap=plt.cm.seismic)
    ax[0, 0].set_title('Phases De Zan')
    ax[0, 1].imshow(np.angle(nl_coh_sqrt), vmin=vmin,
                    vmax=vmax, cmap=plt.cm.seismic)
    ax[0, 1].set_title('Phases Sqrt')
    ax[1, 0].imshow(np.angle(closures.phivec_to_coherence(
        inverted_phases_dezn, n)), vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    ax[1, 0].set_title('De Zan Inverted')
    im = ax[1, 1].imshow(np.angle(closures.phivec_to_coherence(
        inverted_phases_sqrt, n)), vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    ax[1, 1].set_title('Sqrt Inverted')
    plt.colorbar(im,  ax=ax.ravel().tolist(), label='Radians')
    plt.show()

    print('Null Space: ', null_space)

    plt.scatter(sm_difs, np.angle(inverted_phases_dezn), marker='o', color='blue',
                label='De Zan Observed')
    plt.scatter(sm_difs, np.angle(
        closures.coherence_to_phivec(nl_coh_dezan)), marker='x', color='blue', label='De Zan Expected')

    plt.scatter(sm_difs, np.angle(inverted_phases_sqrt),
                marker='o', color='red', label='Sqrt observed')
    plt.scatter(sm_difs, np.angle(closures.coherence_to_phivec(
        nl_coh_sqrt)), marker='x', color='red', label='Sqrt Expected')

    plt.xlabel('sm difference')
    plt.ylabel('Estimated Angle')
    plt.legend(loc='lower left')
    plt.show()

    residual_dezan_ln = closures.coherence_to_phivec(
        nl_coh_dezan) * inverted_phases_dezn.conj()

    residual_sqrt_ln = closures.coherence_to_phivec(
        nl_coh_sqrt) * inverted_phases_sqrt.conj()

    plotn = 6

    fig = plt.figure()

    for i in range(6):

        # PLOT RESIDUALS

        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax[0].imshow(np.angle(
        #     nl_coh_dezan * closures.phivec_to_coherence(inverted_phases_dezn, n).conj()), vmin=vmin, vmax=vmax)
        # ax[0].set_title('Phases De Zan Residual')
        # ax[1].imshow(np.angle(
        #     nl_coh_sqrt * closures.phivec_to_coherence(inverted_phases_sqrt, n).conj()), vmin=vmin, vmax=vmax)
        # ax[1].set_title('Phases Sqrt Residual')
        # plt.show()

        # Do residuals show a systematic relationship to the root soil moisture?

        # plt.plot(sm)
        # plt.show()
        # plt.imshow(sm_to_matrix(sm))
        # plt.show()
        plt.subplot(int(f'23{i+1}'))

        print(null_space.shape)

        random_coeff = np.random.random(
            null_space.shape[0]) * 1 * (i + 0)

        null_vector = np.exp(1j * (null_space.T @ random_coeff))

        norm = np.linalg.norm((null_space.T @ random_coeff), 2)

        residual_dezan = np.angle(residual_dezan_ln * null_vector)

        residual_sqrt = np.angle(residual_sqrt_ln * null_vector)

        # plt.scatter(sm_difs, np.angle(inverted_phases_dezn))
        # plt.scatter(sm_difs, np.angle(closures.coherence_to_phivec(
        #     nl_coh_dezan)))

        # plt.show()

        plt.scatter(sm_difs, residual_dezan,
                    label=f'N(A) vector norm: {np.round(norm, 2)}')
        # plt.scatter(sm_difs, residual_sqrt,
        #             label='Residual Sqrt (rad), sigma = 0.2 rad')
        # plt.plot(sm_difs, np.polyval(
        #     np.polyfit(sm_difs, residual_dezan, 1), sm_difs), '--')
        # plt.plot(sm_difs, np.polyval(
        #     np.polyfit(sm_difs, residual_sqrt, 1), sm_difs), '--')
        plt.legend(loc='upper left')
        plt.xlabel('SM Difference')
        plt.ylabel('Least Norm Residual')
    plt.show()


main()
