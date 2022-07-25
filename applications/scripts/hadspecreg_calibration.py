'''
Created on Dec 16, 2021

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from collections import namedtuple
import os

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix,
    valid_G, hadreg, hadcreg, specreg, enforce_directory, load_object, save_object,
    circular_accuracy)

SimCG0 = namedtuple('SimCG0', ['C_obs', 'G0'])


def accuracy_scenario(hadspecreglparam, data, complex_reg=False):
    if hadspecreglparam is not None:
        hsr = hadspecreglparam
        alpha, nu, beta = (expit(hsr[0]), expit(hsr[1]), expit(hsr[2]))
    acc = []
    for simCG0 in data:
        if hadspecreglparam is not None:
            if not complex_reg:
                G = hadreg(specreg(simCG0.G0.copy(), beta=beta), alpha=alpha, nu=nu)
                C = simCG0.C_obs.copy()
            else:
                C = hadcreg(specreg(simCG0.C_obs.copy(), beta=beta), alpha=alpha, nu=nu)
                G = valid_G(C, corr=True)
        else:
            G = simCG0.G0.copy()
            C = simCG0.C_obs.copy()
        cphases = EMI(C, G=G, corr=False)
        _acc = np.mean(circular_accuracy(cphases))
        acc.append(_acc)
    return np.mean(acc)


def prepare_data(paramlist, rng=None):
    data = []
    for params in paramlist:
        C_obs = correlation(covariance_matrix(decay_model(rng=rng, **params)))
        G0 = valid_G(C_obs, corr=True)
        data.append(SimCG0(C_obs=C_obs, G0=G0))
    return data


def default_paramlist(
        L=100, R=5000, Ps=(40,), coh_decay_list=None, coh_infty_list=None,
        incoh_bad_list=None):
    params0 = {
        'R': R, 'L': L, 'P': 1}
    if coh_decay_list is None: coh_decay_list = [0.5, 0.9]
    if coh_infty_list is None: coh_infty_list = [0.0, 0.2, 0.4]
    if incoh_bad_list is None: incoh_bad_list = [None, 0.0]
    paramlist = []
    for coh_decay in coh_decay_list:
        for coh_infty in coh_infty_list:
            for incoh_bad in incoh_bad_list:
                for P in Ps:
                    params = params0.copy()
                    params['P'] = P
                    params_new = {
                        'coh_decay': coh_decay, 'coh_infty': coh_infty,
                        'incoh_bad': incoh_bad}
                    params.update(params_new)
                    paramlist.append(params)
    return paramlist


def optimize_hadspecreg(
        data, hadspecreglparam0=None, complex_reg=False, maxiter=30, gtol=1e-8):
    if hadspecreglparam0 is None:
        hadspecreglparam0 = np.zeros(3)
    f_noreg = accuracy_scenario(None, data, complex_reg=complex_reg)

    def fun(hadspecreglparam):
        return accuracy_scenario(hadspecreglparam, data, complex_reg=complex_reg)

    options = {'maxiter': maxiter, 'gtol': gtol}
    res = minimize(fun, hadspecreglparam0, method='BFGS', options=options)
    hadregres = {'hadspecreglparam': res.x, 'f': res.fun, 'f_noreg': f_noreg}
    return hadregres


def calibrate_hadspecreg(
        pathout, looks, seed=1, R=10000, Ps=(40,), complex_reg=False,
        coh_decay_list=None, coh_infty_list=None, incoh_bad_list=None,
        overwrite=False, njobs=8, maxiter=20):
    res = {}

    def _calibrate_hadreg(L):
        fnout = os.path.join(pathout, f'{L}.p')
        if overwrite or not os.path.exists(fnout):
            rng = default_rng(seed)
            paramlist = default_paramlist(
                L=L, R=R, Ps=Ps, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list)
            data = prepare_data(paramlist, rng=rng)
            hadregres = optimize_hadspecreg(
                data, complex_reg=complex_reg, maxiter=maxiter)
            save_object(hadregres, fnout)
        else:
            hadregres = load_object(fnout)
        return hadregres

        res[L] = hadregres

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=njobs)(delayed(_calibrate_hadreg)(L) for L in looks)
    for jL, L in enumerate(looks):
        print(L)
        print(expit(res[jL]['hadspecreglparam']), res[jL]['f'], res[jL]['f_noreg'])


def calibrate(path0, njobs=-4, overwrite=False):
    looks = np.arange(4, 21, 1) ** 2
    Ps = (30, 60)
    R = 5000
    scenarios = {
        'broad': (None, None, [None]), 'low': ([0.5], [0.0], [None])}
    rnames = {True: 'G', False: 'complex'}
    for scenario in scenarios:
        for complex_reg in (True , False):
            pathout = os.path.join(path0, rnames[complex_reg], scenario)
            coh_decay_list, coh_infty_list, incoh_bad_list = scenarios[scenario]
            calibrate_hadspecreg(
                pathout, looks, Ps=Ps, R=R, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list,
                complex_reg=complex_reg, njobs=njobs, overwrite=overwrite)

if __name__ == '__main__':
    path0 = '/home2/Work/greg/hadspec'
    calibrate(path0, njobs=9, overwrite=False)

