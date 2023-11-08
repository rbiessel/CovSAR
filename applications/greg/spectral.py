'''
Created on Dec 6, 2021

@author: szwieback
'''
import numpy as np

# Michelli Willoughby
# On Functions Which Preserve the Class of Stieltjes Matrices

# G regularization
# addresses issue that spectrum will tend to be too spread out
# but not always: very small eigenvalues will be overestimated
# hence not clear how well spectral regularization may work

# to compress spectrum: would like it to be concave
# cannot be strictly concave because otherwise will not preserve nonnegativeness
# to see issue: note that the q eigenvector corresponding to the largest ev
# can be chosen to be all positive; subtracting q qT thus reduces all elements
# hence: affine function
# = linear shrinkage
# doi:10.1016/j.csda.2015.09.011 for Bayesian take and adaptive alpha
# doi:10.1016/S0047-259X(03)00096-4 for adaptive [needs 4th order] and theory

# but can have b ** p with 0 < p <= 1 p when M is Stieltjes
# that means that M_ij <= 0 for i!=j


def specreg(G, beta=None, L=None):
    # G can be a Hermitian matrix or just the upper/lower half
    # G is supposed to be a correlation matrix
    # G can also be complex (C_obs)
    P = G.shape[-1]
    if beta is None:
        if L is None: raise ValueError("L or beta need to be provided")
        raise NotImplementedError
    G_out = (1 - beta) * G
    G_out += beta * np.eye(P)[(np.newaxis,) * (len(G.shape) - 2) + (Ellipsis,)]
    return G_out

if __name__ == '__main__':
    pass