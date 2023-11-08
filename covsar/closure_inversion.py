from cv2 import invert, norm
import closures
from scipy import special
from matplotlib import pyplot as plt
import numpy as np
from operator import index
import sys
from sm_forward.sm_forward import SMForward as SM
import library as sarlab
import scipy
from mpl_toolkits import mplot3d


plt.set_loglevel('info')


def grid_search_n(inverted_closures):
    grid_range = 3
    l = 100

    grid = np.linspace(-grid_range, grid_range, l)


# Try a slurry of coefficients to reconstruct a series of phases such that phase closures


def gridded_norm(inverted_closures, null, solution, sm_dif_vec, A):
    gridscale = 3
    xgrid = np.linspace(-gridscale, gridscale, 100)
    ygrid = np.linspace(-gridscale, gridscale, 100)

    X, Y = np.meshgrid(xgrid, ygrid)
    normspace = np.zeros((xgrid.shape[0], ygrid.shape[0]))
    phi_sol = inverted_closures + null @ solution

    # sm triproduct

    S = sm_dif_vec[0] * sm_dif_vec[1] * sm_dif_vec[2]
    closure_angle = A @ inverted_closures
    # closure = beta * S; beta = closure / S

    Beta = closure_angle / S
    print('Beta:', Beta)
    print('S: ', S, Beta * S)
    print('Norm of A:', np.linalg.norm(A, 2))
    print('phi sol norm:', np.linalg.norm(phi_sol, 2))
    print('phi sol norm/S:', np.linalg.norm(phi_sol, 2) / S)
    print('Least norm: ', np.linalg.norm(inverted_closures, 2))
    print('Normalized solution norm:', np.linalg.norm(
        phi_sol, 2) / np.linalg.norm(inverted_closures, 2))
    print('Norm of sm vec: ', np.linalg.norm(sm_dif_vec, 2))

    acceptable_i = []
    acceptable_j = []
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            coeff = np.array([xgrid[i], ygrid[j]]).T
            phis = inverted_closures + null @ coeff
            phism_angle = np.arccos(np.sqrt(
                (phis.T @ sm_dif_vec) / np.sqrt((phis.T @ phis) * (sm_dif_vec.T @ sm_dif_vec))))
            sortfit = np.linalg.norm(np.argsort(
                phis) - np.argsort(sm_dif_vec), 2)
            if sortfit == 0:
                if np.max(np.abs(phis)) <= np.pi/2:
                    acceptable_i.append(xgrid[i])
                    acceptable_j.append(ygrid[j])
                    normspace[j, i] = phism_angle
                # else:
                #     normspace[j, i] = np.nan
            # else:
            #     normspace[j, i] = np.nan

            normspace[j, i] = phism_angle

    normspace[np.where(np.isnan(normspace))] = 1
    sol = np.linalg.norm(phi_sol, 2)
    gamma = np.abs(normspace - sol)
    extent = [-gridscale, gridscale, -gridscale, gridscale]
    plt.pcolormesh(X, Y, normspace, vmin=0, vmax=0.2)
    plt.scatter(np.array(acceptable_i), np.array(
        acceptable_j), marker='.', color='black', s=1)
    plt.axis(extent)
    plt.scatter(solution[0], solution[1], marker='x', color='red')
    print('True solution for phi: ', phi_sol)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, normspace)
    ax.set_xlabel('coeff1')
    ax.set_ylabel('coeff2')
    ax.set_zlabel('Angle')
    ax.scatter(solution[0], solution[1], marker='x', color='red')

    plt.show()

    possible_phis = np.zeros((inverted_closures.shape[0], len(acceptable_j)))
    for i in range(len(acceptable_i)):
        c1 = acceptable_i[i]
        c2 = acceptable_j[i]
        possible_phis[:, i] = inverted_closures + null @ np.array([c1, c2])
    return possible_phis


def main():
    n = 3
    # definte the time steps
    t = np.linspace(0, 1, n)

    # define a displacement time-series
    displacement = np.linspace(0, 1, n)

    sm = np.abs(np.random.randn(n) * 10) + 25
    print('sm: ', sm)

    forward_model = SM(
        imag_slope=1/10, r_A=0.01, r_B=0.1, r_C=4)

    forward_model.set_moistures(sm)
    sm = sm[:, np.newaxis]
    # plt.plot(t, smd)

    # Simulate

    def ts2cov(displacement, soil_moisture):
        coherence = np.ones(
            (len(displacement), len(displacement)), dtype=np.complex64)
        for i in range(len(displacement)):
            for j in range(len(displacement)):
                displacement_angle = np.exp(
                    displacement[j] * 1j) * np.exp(displacement[i]*1j).conj()
                non_lin_contrib = forward_model.get_phases_dezan(
                    soil_moisture[j], soil_moisture[i])
                coherence[i, j] = displacement_angle * \
                    np.exp(np.angle(non_lin_contrib) * 1j)
        return coherence

    sm_dif = np.tile(sm, (sm.shape[0]))
    sm_dif_normed = (sm_dif.T - sm_dif)  # / (sm_dif.T + sm_dif)

    coherence_sm = ts2cov(displacement, sm)
    coherence_disp = ts2cov(displacement, np.zeros(displacement.shape))
    sm_component = ts2cov(displacement * 0, sm)

    k = special.comb(coherence_sm.shape[0] - 1, 2)
    triplets = closures.get_triplets(coherence_sm.shape[0])
    triplets = triplets[0:int(k)]

    indexes = closures.collapse_indexes(coherence_sm)
    A = closures.build_A(triplets, indexes)

    triu_indexes = np.triu_indices(coherence_sm.shape[0], 1)
    phi_vec = coherence_sm[triu_indexes]
    sm_dif_vec = sm_dif_normed[triu_indexes]

    phi_vec_observed = phi_vec
    phi_expected_vec = coherence_disp[triu_indexes]

    phi_expected_sm = sm_component[triu_indexes]

    for i in range(1):

        xis = A @ np.angle(phi_vec)

        null = scipy.linalg.null_space(A)
        print('null space shape: ', null.shape)

        inverted_closures = A.T @ np.linalg.inv(A @ A.T) @ xis

        print('input phi: ', np.angle(phi_vec))
        print('least norm phi vector: ', inverted_closures)
        # Find the coefficients needed to shift the least norm solution to the actual solution
        x = np.linalg.lstsq(null, np.angle(
            phi_expected_sm) - inverted_closures)[0]

        reconstructed = inverted_closures + null @ x
        print(' Coefficients for true solution: ', x)

        possibles = gridded_norm(inverted_closures, null, x, sm_dif_vec, A)
        possible_est = np.exp(np.mean(possibles, axis=1) * 1j)
        possible_std = np.exp(1 * np.std(possibles, axis=1) * 1j)

        inverted_closures = np.exp(1j * inverted_closures)
        reconstructed_phases = np.exp(1j * reconstructed)
        phi_vec = phi_vec * np.conj(inverted_closures)
        plt.plot(np.angle(phi_vec),
                 '-', label=f'Estimate LN {i}')

    ind = np.linspace(0, len(phi_expected_vec) - 1, len(phi_expected_vec))

    plt.plot(ind, np.angle(phi_vec_observed), label='Observed')
    plt.plot(ind, np.angle(phi_expected_vec), label='Target')

    # Plot estimated phase vector
    # mean
    plt.plot(ind, np.angle(phi_vec_observed * possible_est.conj()), '--',
             label='estimated grid')
    # # 1 stdv
    top = np.angle(phi_vec_observed * possible_est.conj() * possible_std)
    lower = np.angle(phi_vec_observed * possible_est.conj()
                     * possible_std.conj())
    plt.fill_between(ind, y1=top, y2=lower, alpha=0.1, color='red')

    plt.legend(loc='lower left')
    plt.show()
    plt.scatter(np.abs(sm_dif_vec), np.abs(
        np.angle(possible_est)), label='gs prediction')
    plt.scatter(np.abs(sm_dif_vec), np.abs(
        np.angle(phi_expected_sm)), label='target')
    plt.xlim(left=0)
    plt.legend(loc='lower right')
    plt.xlabel('SM dif')
    plt.ylabel('Phase difference')
    plt.show()

    # Compute raw time series:
    return
    disp_timeseries = sarlab.eig_decomp(
        coherence_disp[:, :, np.newaxis, np.newaxis])

    corrected_timeseries = sarlab.eig_decomp(
        (phi_vec_observed * possible_est.conj())[:, :, np.newaxis, np.newaxis])

    plt.plot(disp_timeseries[0, 0])
    plt.plot(corrected_timeseries[0, 0])
    plt.show()


main()
