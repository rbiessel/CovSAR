import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.twodim_base import tri
from numpy.random import permutation
from dielectric import mv2eps_real, mv2eps_imag
from skimage.util import random_noise
import itertools

omega = 5.405e9  # Hz
# omega = 1e9


def get_adjacent_triplets(num):

    triplets = []
    numbers = np.arange(0, num, 1)
    for i in numbers:
        indices = range(i, i+3)
        triplet = numbers.take(indices, axis=0, mode='wrap')
        triplets.append(triplet)

    return np.sort(np.array(triplets))


def is_increasing(triplet, wrap=True):

    if triplet[2] > triplet[1] > triplet[0]:
        return True

    if wrap:
        triplet = np.roll(triplet, 1)
        if triplet[2] > triplet[1] > triplet[0]:
            return True

        triplet = np.roll(triplet, 1)
        if triplet[2] > triplet[1] > triplet[0]:
            return True

    return False


def get_triplets(num, wrap=True):
    '''
        Get all possible interferogram triplets by index
    '''
    numbers = np.arange(0, num, 1)
    permutations = np.array(list(itertools.permutations(numbers, 3)))
    permutations = np.unique(permutations, axis=1)

    permutations = [
        permutation for permutation in permutations if is_increasing(permutation, wrap=wrap)]

    return permutations


def k_prime(epsilon: complex) -> complex:
    μ0 = 1.25663706212e-6  # magnetic permeability of free space
    eps0 = 8.854187e-12  # electric permittivity of free space
    epsilon = epsilon * eps0  # convert relative permittivity to absolute

    return -np.sqrt(omega**2 * epsilon * μ0)


def sm2eps(sm) -> np.complex:

    complex_array = np.zeros(sm.shape, dtype=np.complex64)
    complex_array += mv2eps_real(sm)
    complex_array += (mv2eps_imag(sm) * 1j)

    return complex_array


def get_phase_dezan(stack, ref: int, sec: int):
    '''
        Convert a pair of soil-moisture maps into an interferogram via an analytical forward model
        Adapted from De Zan et al., 2014
    '''
    eps1 = sm2eps(stack[ref])
    eps2 = sm2eps(stack[sec])

    k1 = k_prime(eps1)
    k2 = k_prime(eps2)

    phi = (2j * np.sqrt(k2.imag * k1.imag)) / (k2.conj() - k1)

    dif = stack[ref] - stack[sec]

    # plt.plot(dif.flatten(), np.angle(phi).flatten(), '.')
    # plt.xlabel('Change in Soil Moisture (mv)')
    # plt.ylabel('Phase Change')
    # plt.show()

    return phi


def z_func(x, y):
    return (x**2+y**3) * np.exp(-(x**2+y**2))


def z_linear(x, y):
    return (x + y)


# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def main():
    x = np.arange(-5.0, 5.0, 0.05)
    y = np.arange(-5.0, 5.0, 0.05)

    tm = [0.8, 0.6, 0.03, 0.99, 0.1]
    tm = [0.1, 0.5, 0.7]
    stack = np.ones((len(tm), len(x), len(y)))
    rand = np.random.rand(len(x), len(y))

    for i in range(len(tm)):
        X, Y = np.meshgrid(x, y)  # grid of point
        Z = z_func(X, Y)  # evaluation of the function on the grid
        # Z = Z * tm[i]
        Z = gaus2d(x=X, y=Y, sx=2, sy=2)
        Z = random_noise(Z, mode='speckle', var=0.02**2)

        Z = (Z / Z.max()) * 50 * tm[i]

        # Z2 = gaus2d(x=X, y=Y, mx=5, my=5, sx=2, sy=2)
        # Z2 = random_noise(Z2, var=0)
        # Z2 = (Z2 / Z2.max()) * 50 * tm2[i]

        # Z[rand < 0.5] = Z2[rand < 0.5]

        stack[i] = Z

    # for mv in stack:
    #     plt.imshow(mv)
    #     plt.show()

    # if stack.min() < 0:
    #     stack = stack + -1 * stack.min()
    # stack = stack / stack.max()
    # stack *= 50

    triplets = get_adjacent_triplets(len(tm))
    triplets = get_triplets(len(tm), wrap=False)

    print(triplets)
    # triplets = [triplet for triplet in triplets if i in triplet]

    # triplets = [[0, 1, 2], [1, 2, 0], [2, 0, 1],
    #             [1, 0, 2], [0, 2, 1], [2, 1, 0]]

    tripstack = np.zeros(
        (len(triplets), stack.shape[1], stack.shape[2]), dtype=np.complex64)

    for t in range(len(tripstack)):
        triplet = triplets[t]
        phi12 = get_phase_dezan(stack, triplet[0], triplet[1])
        phi23 = get_phase_dezan(stack, triplet[1], triplet[2])
        phi13 = get_phase_dezan(stack, triplet[0], triplet[2])

        closure = phi12 * phi23 * phi13.conj()
        tripstack[t] = closure

    fig, ax = plt.subplots(nrows=2, ncols=1)
    for i in range(len(tripstack)):
        closure = tripstack[i]

        data = closure[closure != 0]
        # all = closure_lc[closure_lc != 0]

        ax[0].scatter(np.abs(data),
                      np.angle(data), s=1, label=triplets[i])
        ax[0].legend(loc='lower right')
        ax[0].set_xscale('log')
        ax[0].set_ylim(-np.pi/2, np.pi/2)
        ax[0].set_xlabel('Bicoherence')
        ax[0].set_ylabel('Phase Closure')

    xs = np.arange(0, len(tm), 1)
    ax[1].plot(xs, tm)
    ax[1].set_ylabel('mv')
    ax[1].set_xlabel('time')

    plt.show()


main()
