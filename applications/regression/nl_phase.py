import numpy as np
import numba as nb
from matplotlib import pyplot as plt
import time


def jacobian(s, x):
    '''
        Compute the gradient of the residual at s
    '''
    return x.flatten() * (np.sin(s * x.flatten()) - np.cos(s * x.flatten()))


def jacobian2(s, x):
    '''
        Compute the gradient of the residual at s
    '''
    return x.flatten()


def r(phases, x, s, angle=True, norm=False):
    '''
        Compute residual using either the argument of the complex numbers (angle=True)
        or the real and imaginary parts of the observed phase (angle=False)
    '''

    if angle:
        r = np.angle(np.exp(1j * s * x.flatten()) * phases.flatten().conj())
    else:
        r = (phases.real.flatten() - np.cos(s * x.flatten())) + \
            (phases.imag.flatten() - np.sin(s * x.flatten()))
    if norm:
        return np.linalg.norm(r, 1)
    else:
        return r


def grid_search(phases, x, s=0, rnge=2, N=10, rr=False):
    '''
        Use a grid search to optimize the frequency s

        rr: return array of residuals as well as optimal s
    '''

    grid = np.linspace(s - rnge, s + rnge, N)
    residuals = np.zeros(grid.shape)

    for i in range(grid.shape[0]):
        residuals[i] = r(phases, x, grid[i], angle=True, norm=True)

    if rr:
        return grid[np.argmin(residuals)], grid, residuals
    else:
        return grid[np.argmin(residuals)]


def grad_descent(phases, x, s, maxi=50):
    m = s
    eta = 1

    for i in range(maxi):
        mprev = m
        J = jacobian2(m, x).T
        res = r(phases, x, m, angle=True)
        inv = 1/(J.T @ J)
        m = m - eta * ((inv * J.T) @ res)
        if np.abs((m - mprev)/m) < 0.001:
            # print(f'Finished Inversion with {i} iterations.')
            return m, i

    return m, maxi


def estimate_s(phases, x, maxi=50, gridN=5, s0=0, rnge=2, gradDescent=True):
    '''
        Full workflow to estiamte s
        phases: phases
        maxi: maximum number of iterations for gradient descent
        gridN
    '''
    m, grid, rr = grid_search(phases, x, s=s0, rnge=rnge, N=gridN, rr=True)
    if gradDescent:
        m, iter = grad_descent(phases, x, m, maxi=maxi)
        return m, iter
    else:
        return m, grid, rr


def simulate(N=20, sigma=1):

    N = 20
    sm = np.random.normal(loc=30, scale=2, size=N)

    s = np.random.normal(loc=0, scale=1, size=1)[0]
    s = 1.32

    intensities = np.tile(sm, (len(sm), 1))
    intensities = (intensities.T - intensities)

    phase = np.exp(1j * intensities * s)
    noise = np.exp(1j * np.random.normal(loc=0,
                                         scale=sigma, size=intensities.shape))
    phase = phase * noise

    return intensities, phase, s


def main():
    errors = np.zeros(10)
    for i in range(errors.shape[0]):
        intenities, phases, s = simulate(sigma=1)
        n = 5
        m, grid, residuals = grid_search(
            phases, intenities, s=0, rnge=2, N=1000, rr=True)
        m0 = grid_search(phases, intenities, s=0, rnge=2, N=n)

        m, iter = grad_descent(phases, intenities, m0, maxi=50)
        errors[i] = s - m

        coeff = np.polyfit(intenities.flatten(), np.angle(phases).flatten(), 2)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax[0].set_title('Simulated Data & Model')
        ax[0].scatter(intenities.flatten(), np.angle(
            phases), s=10, color='black', alpha=0.7)
        ax[0].set_title(f'Iterations: {iter}, Grid Search N: {n}')
        ax[0].plot(intenities.flatten(), intenities.flatten()
                   * s, '--', label='Truth', color='tomato', linewidth=4, dashes=(4, 2))
        ax[0].plot(intenities.flatten(), intenities.flatten()
                   * m, label='Gauss Newton Est.', color='steelblue')
        ax[0].plot(intenities.flatten(), intenities.flatten()
                   * m0, label='Grid Search Est.', color='gray', alpha=0.5)

        ax[0].plot(intenities.flatten(), intenities.flatten()
                   * coeff[0], '--', label='OLS Est.', color='darkgoldenrod', alpha=0.8)

        ax[0].set_ylim((-5, 6))
        ax[0].legend(loc='best')
        ax[0].set_ylabel('Phase (rad)')
        ax[0].set_xlabel('Log Intensity Difference (dB)')

        ax[1].set_title('Misfit')
        ax[1].plot(grid, residuals, color='black', alpha=0.8, linewidth=2)
        ax[1].axvline(s, color='tomato', label='Model Truth',
                      alpha=0.8, linewidth=4, linestyle='--')
        ax[1].axvline(m, color='steelblue', label='Estimate', alpha=0.8)

        ax[1].set_xlabel('Parameter $s$')
        ax[1].set_ylabel('Residual $|r|$')
        plt.show()

    plt.hist(errors, bins=100)
    plt.show()


if __name__ == "__main__":
    main()
