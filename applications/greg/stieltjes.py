'''
Created on Dec 2, 2021

@author: szwieback
'''

import numpy as np

    
def approximate_stieltjes(G_raw, min_eig_frac=1e-3):
    # G_raw is symmetric and nonnegative, but not necessarily positive def.
    # nonnegativeness is enforced
    # returns M, an approximate Stieltjes inverse of G_raw
    from numpy.linalg import eigh
    P = G_raw.shape[0]
    lam, U = eigh(np.abs(G_raw))
    lam_crit = np.abs(lam[-1]) * min_eig_frac
    lam[lam < lam_crit] = lam_crit
    M_trial = np.matmul(np.power(lam, -1)[np.newaxis,:] * U, U.T)
    M_trial_diag = np.diag(M_trial).copy()
    np.fill_diagonal(M_trial, 0)
    offset = max((0, np.max(M_trial)))
    np.fill_diagonal(M_trial, M_trial_diag + offset * P)
    M_trial -= offset * np.ones((P, P))
    return M_trial
    
    # G_out = G_raw.copy()
    # lam = eigh(
    #     G_raw, subset_by_index=[0, 0], eigvals_only=True)
    # min_eig = min_eig_frac * np.trace(G_raw)
    # negshift = lam[0] - min_eig
    # if negshift < 0:
    #     G_out -= negshift * np.eye(P)
    # g_min = min((0, np.min(G_out)))
    # G_out += g_min
    

def newton_backtracking_line_search(
        x, dx, f, dec2, alpha=0.1, beta=0.5, max_iter=100):
    t = 1.0
    found = False
    n = 1
    f_x = f(x)
    while not found:
        x_trial = x + t * dx
        df_trial = f(x_trial) - f_x
        df_bound = -alpha * t * dec2
        if np.isfinite(df_trial) and df_trial <= df_bound:
            found = True
        else:
            t *= beta
        n += 1
        if n >= max_iter:
            raise RuntimeError("Backtracking line search did not converge")
    return x_trial


def newton(x0, f, grad_f, hess_f, epsilon=1e-6, max_steps=25):
    from scipy.linalg import solve
    x = x0
    
    for step in range(max_steps):
        H = hess_f(x)
        g = grad_f(x)
        dx = solve(H, -g, assume_a='pos', check_finite=False)
        dec2 = -np.dot(g, dx)
        if dec2 < 2 * epsilon:
            return x
        else:
            x = newton_backtracking_line_search(x, dx, f, dec2,)
        print(step, f(x), dec2)


def grad_cp(t, x, P, c, M_inv, ind):
    g = t * c
    g[:P] -= t * np.diag(M_inv)
    g[P:] -= 2 * t * M_inv[ind[0, P:], ind[1, P:]]
    g[P:] -= x[P:] ** (-1)
    return g

def _M_i_test(i, P, ind):
    M_i = np.zeros((P, P))
    if i < P:
        M_i[i, i] = 1
    else:
        M_i[ind[0, i], ind[1, i]] = 1
        M_i[ind[1, i], ind[0, i]] = 1
    return M_i

def _F_i_test(i, P, ind):
    N_M = ind.shape[1]
    F_i = np.zeros((N_M - P, N_M - P))
    if i >= P:
        F_i[i - P, i - P] = -1
    return F_i

def _F_inv_test(x, P, ind):
    F = np.zeros((N_M - P, N_M - P))
    for i in range(ind.shape[1]):
        F += x[i] * _F_i_test(i, P, ind)
    return np.linalg.inv(F)

def _grad_cp_test(t, x, P, c, M_inv, ind):
    g = t * c
    F_inv = _F_inv_test(x, P, ind)
    for i in range(ind.shape[1]):
        g[i] -= t * np.trace(np.matmul(M_inv, _M_i_test(i, P, ind)))
        g[i] -= np.trace(
            np.matmul(F_inv, _F_i_test(i, P, ind)))
    return g

def _hess_cp_test(t, x, P, M_inv, ind):
    N_M = ind.shape[1]
    H = np.zeros((N_M, N_M))
    F_inv = _F_inv_test(x, P, ind)
    for i1 in range(N_M):
        F_i1, M_i1 = _F_i_test(i1, P, ind), _M_i_test(i1, P, ind)
        for i2 in range(N_M):
            F_i2, M_i2 = _F_i_test(i2, P, ind), _M_i_test(i2, P, ind)
            tr_F = np.trace(
                np.matmul(np.matmul(F_inv, F_i1), np.matmul(F_inv, F_i2)))
            tr_M = np.trace(
                np.matmul(np.matmul(M_inv, M_i1), np.matmul(M_inv, M_i2)))
            H[i1, i2]= t * tr_M + tr_F
    return H
    

def _N_M(P):
    return (P * (P + 1)) // 2


def M_inv_det(x, ind, P):
    M = np.zeros((P, P))
    M[ind[0, ...], ind[1, ...]] = x
    M[ind[1, P:], ind[0, P:]] = x[P:]  # little harm in also providing upper half
    from scipy.linalg import cho_factor, cho_solve
    lower = False
    cho = cho_factor(M, lower=lower, check_finite=False)
    M_inv = cho_solve(cho, np.eye(P))
    M_det = np.product(np.diag(cho[0]))   
    return M, M_inv, M_det


def indices(P):
    ind = np.zeros((2, _N_M(P)), dtype=np.uint32)
    ind[:,:P] = np.arange(P)[np.newaxis,:]
    k = 0
    for l1 in range(P):
        for l2 in range(l1):
            ind[:, P + k] = np.array([l1, l2])
            k += 1 
    return ind


def test_newton():
    N = 500
    rng = np.random.default_rng(seed=1)
    L = rng.standard_normal((5 * N, N))
    A = np.dot(L.T, L)
    b = rng.standard_normal(N)
    
    def f(x):
        return 0.5 * np.dot(x, np.dot(A, x)) + np.exp(np.dot(b, x))
    
    def grad_f(x):
        return np.dot(A, x) + np.exp(np.dot(b, x)) * b
    
    def hess_f(x):
        return A + np.exp(np.dot(b, x)) * np.outer(b, b)
    
    x0 = -b / N
    x = newton(x0, f, grad_f, hess_f, epsilon=1e-10)
    return x

    
if __name__ == '__main__':
    # test_newton()
    P = 16
    ind = indices(P)
    N_M = _N_M(P)
    
    G_0 = np.ones((P, P))
    G_0 += (np.diag(np.arange(P) + 3 * np.ones(P)))
    M_0 = np.linalg.inv(G_0)
    x_0 = M_0[ind[0,:], ind[1,:]]
    
    c = np.ones((N_M))
    M, M_inv, M_det = M_inv_det(x_0, ind, P)
    
    t = 100
    x = x_0
    dg = grad_cp(t, x, P, c, M_inv, ind) - _grad_cp_test(t, x, P, c, M_inv, ind)
    
    def f(x):
        try:
            M, M_inv, M_det = M_inv_det(x, ind, P)
            fx = np.dot(c, x) - np.log(M_det)
            fgx = grad_cp(t, x, P, c, M_inv, ind)
        except:
            fx = np.infty
            fgx = t * c
        return fx, fgx
    
    
    