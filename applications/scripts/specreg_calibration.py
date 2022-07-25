'''
Created on Dec 7, 2021

@author: szwieback
'''

from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from collections import namedtuple
import os

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix,
    valid_G, specreg, enforce_directory, load_object, save_object,
    circular_accuracy)

from hadreg_calibration import default_paramlist, prepare_data

SimCG0 = namedtuple('SimCG0', ['C_obs', 'G0'])

def accuracy_scenario(specregparam, data, complex_reg=False):
    if specregparam is not None:
        beta = expit(specregparam)
    acc = []
    for simCG0 in data:
        if specregparam is not None and complex_reg:
            C = specreg(simCG0.C_obs.copy(), beta=beta)
            G = valid_G(C.copy(), corr=True)
        elif specregparam is not None and not complex_reg:
            G = specreg(simCG0.G0.copy(), beta=beta)
            C = simCG0.C_obs.copy()
        else:
            G = simCG0.G0.copy()
            C = simCG0.C_obs.copy()
        cphases = EMI(C, G=G, corr=False)
        _acc = np.mean(circular_accuracy(cphases))
        acc.append(_acc)
    return np.mean(acc)

def optimize_specreg(
        data, specregparam0=None, complex_reg=False, maxiter=20, gtol=1e-8):
    if specregparam0 is None:
        specregparam0 = 0.0
    f_noreg = accuracy_scenario(None, data, complex_reg=complex_reg)

    def fun(specregparam):
        return accuracy_scenario(specregparam, data, complex_reg=complex_reg)

    options = {'maxiter': maxiter, 'gtol': gtol}
    res = minimize(fun, specregparam0, method='BFGS', options=options)
    specregres = {'specregparam': res.x, 'f': res.fun, 'f_noreg': f_noreg}
    return specregres


def calibrate_specreg(
        pathout, looks, seed=1, R=10000, Ps=(40,), complex_reg=False,
        coh_decay_list=None, coh_infty_list=None, incoh_bad_list=None,
        overwrite=False, njobs=-3, maxiter=20):
    res = {}

    def _calibrate_specreg(L):
        fnout = os.path.join(pathout, f'{L}.p')
        if overwrite or not os.path.exists(fnout):
            rng = default_rng(seed)
            paramlist = default_paramlist(
                L=L, R=R, Ps=Ps, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list)
            data = prepare_data(paramlist, rng=rng)
            specregres = optimize_specreg(
                data, complex_reg=complex_reg, maxiter=maxiter)
            save_object(specregres, fnout)
        else:
            specregres = load_object(fnout)
        return specregres

        res[L] = specregres

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=njobs)(delayed(_calibrate_specreg)(L) for L in looks)
    for jL, L in enumerate(looks):
        print(L)
        print(expit(res[jL]['specregparam']), res[jL]['f'], res[jL]['f_noreg'])


def calibrate(path0, njobs=-3, overwrite=False):
    looks = np.arange(4, 21, 1) ** 2
    Ps = (30, 60)
    R = 5000
    scenarios = {
        'broad': (None, None, [None]), 'low': ([0.5], [0.0], [None])}
    rnames = {True: 'G', False: 'complex'}
    for scenario in scenarios:  # scenarios
        for complex_reg in (True , False):
            pathout = os.path.join(path0, rnames[complex_reg], scenario)
            coh_decay_list, coh_infty_list, incoh_bad_list = scenarios[scenario]
            calibrate_specreg(
                pathout, looks, Ps=Ps, R=R, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list,
                complex_reg=complex_reg, njobs=njobs, overwrite=overwrite)


if __name__ == '__main__':
    path0 = '/home2/Work/greg/spectral'
    calibrate(path0, njobs=9, overwrite=True)

