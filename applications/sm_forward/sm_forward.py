import numpy as np
from matplotlib import pyplot as plt


class SMForward:
    '''
        Object for performing soil-moisture phase contribution simulations based on sensitivity of soil dielectric properties to soil moisture
        From De Zan et a., 2015
    '''
    mvs: np.array = None
    de_real = None
    de_imag = None

    def __init__(self, imag_slope, r_A, r_B, r_C, omega=5.405e9):

        self.imag_slope = imag_slope
        self.real_A = r_A
        self.real_B = r_B
        self.real_C = r_C

        self.omega = omega

    def mv2eps_real(self, sm):
        return self.real_C + self.real_B * sm + self.real_A * sm**2

    def mv2eps_imag(self, sm):
        return sm * self.imag_slope

    def set_moistures(self, mvs):
        self.mvs = mvs
        self.de_real = self.mv2eps_real(self.mvs)
        self.de_imag = self.mv2eps_imag(self.mvs)  # self.mvs * self.imag_slope

    def plot_dielectric(self):

        plt.scatter(self.mvs, self.de_real, label='Real Part')
        plt.scatter(self.mvs, self.de_imag, label='imaginary part')
        plt.legend(loc='lower left')
        plt.show()

    def sm2eps(self, sm) -> np.complex:
        complex_array = np.zeros(sm.shape, dtype=np.complex64)
        complex_array += self.mv2eps_real(sm)
        complex_array += (self.mv2eps_imag(sm) * 1j)
        return complex_array

    def k_prime(self, epsilon: complex) -> complex:
        μ0 = 1.25663706212e-6  # magnetic permeability of free space
        eps0 = 8.854187e-12  # electric permittivity of free space
        epsilon = epsilon * eps0  # convert relative permittivity to absolute
        return -np.sqrt(self.omega**2 * epsilon * μ0)

    def I_dezan(self, sm):
        eps1 = self.sm2eps(sm)
        k1 = self.k_prime(eps1)
        return np.abs(1/((2j * k1) - (2j * k1.conj())))**2

    def fresnels(self, sm, theta=np.radians(45)):
        '''
            Compute reflection coefficient according to some sm
        '''
        epsT = self.sm2eps(sm).real
        epsI = 1
        return np.cos(theta) * ((epsT - epsI) / (epsT + epsI))

    def reflected_I(self, sm, theta):
        n1 = 1
        eps = self.sm2eps(sm).real
        alpha = np.sqrt(1 - ((n1/eps) * np.sin(theta))**2) / np.cos(theta)
        beta = eps/n1
        return (alpha - beta) / (alpha + beta)

    def dubois_I_ratio(self, sm1, sm2, theta=10, log=False):
        ''''
            Returns the expected ratio of intensity1/intensity2 corresponding to a change in dielectric constant

        '''

        eps1 = self.mv2eps_real(sm1)
        eps2 = self.mv2eps_real(sm2)

        ratio = 10**(eps1)/10**(eps2)
        if log:
            return np.log10(ratio)
        else:
            return ratio

    def get_phases_dezan(self, ref, sec, use_epsilon=False):
        '''
            Convert a pair of soil-moisture maps into an interferogram via an analytical forward model
            Adapted from De Zan et al., 2014
        '''
        if not use_epsilon:
            eps1 = self.sm2eps(ref)
            eps2 = self.sm2eps(sec)
        else:
            eps1 = ref
            eps2 = sec

        k1 = self.k_prime(eps1)
        k2 = self.k_prime(eps2)

        phi = (2j * np.sqrt(k2.imag * k1.imag)) / (k2.conj() - k1)
        # gamma = 1 / ((2j * k1) - (2j * k2.conj()))
        # return gamma
        return np.nan_to_num(phi)
