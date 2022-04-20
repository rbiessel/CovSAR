import numpy as np
from matplotlib import pyplot as plt


def least_norm(A, closures, pinv=False):
    '''
        Solve: Ax = b
        Find the minimum norm vector 'x' of phases that can explain 'b' phase closures
    '''
    if pinv:
        return np.linalg.pinv(A) @ closures
    return A.T @ np.linalg.inv(A @ A.T) @ closures


def get_closures(A, phi):
    '''
        Use matrix mult to generate vector of phase closures such that 
        the angle xi = phi_12 + phi_23 - phi_13
    '''
    return A @ np.angle(phi)


# Vector of phases with zero phase closure errors
closed = np.exp(1j * np.array([0.5, 1, 0.5]))

# Vector of phases with non-zero closure errors
unclosed = np.exp(1j * np.array([0.5, 1 + 0.1, 0.5 + 0.1]))


def main(phis):
    ind = np.array([1, 2, 3])
    A = np.array([[1, -1, 1]])

    # Phases to phase closures
    closures = get_closures(A, phis)

    # Phase closures back to phases
    least_norm_phi = least_norm(A, closures)
    print('Phase closures: ', closures)

    plt.plot(ind, np.angle(closed), label='observed phases')
    plt.plot(ind, np.angle(least_norm_phi), label='least norm phases ')
    plt.plot(ind, np.angle(closed * least_norm_phi.conj()),
             label='Corrected Phases')

    plt.legend(loc='lower left')
    plt.show()


main(unclosed)
