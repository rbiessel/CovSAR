import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.twodim_base import tri
from numpy.random import permutation
from dielectric import mv2eps_real, mv2eps_imag
from skimage.util import random_noise
import itertools

omega = 5.405e9  # Hz


def get_triplets(num):
    '''
        Get all possible interferogram triplets by index
    '''
    numbers = np.arange(0, num, 1)
    permutations = np.array(list(itertools.permutations(numbers, 3)))
    permutations = np.unique(permutations, axis=1)

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


def main():
    return
    nlparameter = 1/2

    soil_moisture = np.array([0.2, 0.3, 0.4, 0.2, 0.1, 0.5, 0.3])
    phi01 = get_phase_dezan(soil_moisture, 0, 1)
    phi12 = get_phase_dezan(soil_moisture, 1, 2)
    phi23 = get_phase_dezan(soil_moisture, 2, 3)
    phi34 = get_phase_dezan(soil_moisture, 3, 4)

    phases = np.array([phi01, phi12, phi23, phi34], dtype=np.complex64)
    timeseries = np.cumprod(phases)
    # plt.plot(np.angle(timeseries))
    # plt.show()

    triplets = get_triplets(len(soil_moisture))

    for param in np.linspace(0, 5, 25):
        closures = []
        amptriplets = []
        for triplet in triplets:
            phi12 = get_phase_dezan(soil_moisture, triplet[0], triplet[1])
            phi23 = get_phase_dezan(soil_moisture, triplet[1], triplet[2])
            phi13 = get_phase_dezan(soil_moisture, triplet[0], triplet[2])

            closure = phi12 * phi23 * phi13.conj()
            closures.append(np.angle(closure))

            nlfactor = param

            sm = soil_moisture * 3
            amptriplet = (sm[triplet[1]] - sm[triplet[0]])**nlfactor + (sm[triplet[2]] -
                                                                        sm[triplet[1]])**nlfactor - (sm[triplet[2]] - sm[triplet[0]])**nlfactor

            amptriplet = sm[triplet[0]]**nlfactor + \
                sm[triplet[1]]**nlfactor - sm[triplet[2]]**nlfactor
            amptriplets.append(amptriplet)

        # plt.scatter(closures, amptriplets)
        # plt.title(f'param {param}')
        # plt.show()


main()
