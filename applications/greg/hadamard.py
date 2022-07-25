'''
Created on Oct 25, 2021

@author: simon
'''

import numpy as np
from greg.preproc import force_doubly_nonnegative, correlation, valid_G

def hadreg(G, theta=(1.0,), alpha=None, nu=None, L=None):
    # G can be a Hermitian matrix or just the upper/lower half
    # G needs to be a correlation G
    G_out = np.zeros_like(G)
    if L is not None and theta is None and (alpha is None or nu is None):
        raise NotImplementedError
    if L is None and alpha is not None and nu is not None:
        theta = theta_from_alpha_nu(alpha, nu)
    for jtheta, thetaj in enumerate(theta):
        if thetaj < 0: raise ValueError(f"{thetaj} less than zero")
        G_out += thetaj * (G ** (jtheta + 1))
    return G_out

def hadspecreg(G, alpha=None, nu=None, beta=None, L=None):
    from greg.spectral import specreg
    if L is not None and (alpha is None or nu is None or beta is None):
        raise NotImplementedError
    G_out = hadreg(specreg(G, beta=beta), alpha=alpha, nu=nu)
    return G_out

def hadcreg(C_obs, G=None, theta=(1.0,), alpha=None, nu=None, L=None):
    # G can be a Hermitian matrix or just the upper/lower half
    Cr_obs = C_obs.copy()
    hadm = Cr_obs.copy()
    if L is not None and theta is None and (alpha is None or nu is None):
        raise NotImplementedError
    if L is None and alpha is not None and nu is not None:
        theta = theta_from_alpha_nu(alpha, nu)
    if G is None:
        G = valid_G(Cr_obs)
    for jtheta, thetaj in enumerate(theta):
        if thetaj < 0: raise ValueError(f"{thetaj} less than zero")
        hadm  += thetaj * (G ** (jtheta))
    Cr_obs *= hadm
    return Cr_obs

def theta_from_alpha_nu(alpha, nu):
    # alpha and nu between 0 and 1
    # f'(0) = 1 - alpha
    # f'(1) = 1 + alpha (1 + nu)
    theta = np.array([1 - alpha, nu * alpha, (1 - nu) * alpha])
    return theta
