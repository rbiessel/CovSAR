'''
Created on Feb 17, 2022

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from collections import namedtuple
import os
from itertools import product as cartprod

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix,
    valid_G, regularize_G, enforce_directory, load_object, save_object,
    circular_accuracy, read_parameters)

rtypes_def = ['hadamard', 'spectral', 'hadspec', 'none']

def synthetic_scenario_coh(
        coh_decay_t, coh_infty_t, rtypes=None, L=40, P=30, R=5000, incoh_bad=0.0,
        complex_reg=False, seed=1, fnout=None, overwrite=False):
    if rtypes is None: rtypes = rtypes_def
    if complex_reg: raise NotImplementedError
    if fnout is not None and os.path.exists(fnout) and not overwrite:
        acc = load_object(fnout)
    else:
        rng = default_rng(seed)
        coh_parms = list(cartprod(coh_decay_t, coh_infty_t))
        acc = {rtype: [] for rtype in rtypes}
        acc['decay_infty']= coh_parms
        for coh_decay, coh_infty in coh_parms:
            y = decay_model(
                    R=R, L=L, P=P, coh_decay=coh_decay, coh_infty=coh_infty,
                    incoh_bad=incoh_bad, rng=rng)
            C_obs = correlation(covariance_matrix(y))
            G0 = valid_G(C_obs, corr=True)
            for rtype in rtypes:
                G = regularize_G(G0, rtype, **params_rtypes[rtype])
                cphases = EMI(C_obs.copy(), G=G, corr=False)
                _acc = np.mean(circular_accuracy(cphases))
                acc[rtype].append(_acc)
        if fnout is not None:
            save_object(acc, fnout)
    return acc

def plot_synthetic(acc, fnfig):
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, colsbg
    from string import ascii_lowercase
    np.set_printoptions(precision=2)
    rtypes_plot = ['hadamard', 'spectral', 'hadspec']
    labels = {'hadamard': 'Hadamard', 'spectral': 'Spectral', 
              'hadspec': 'Hadamard--spectral'}
    decay_infty_plot = [(coh_decay, None) for coh_decay in coh_decay_t]
    di_all = np.array(acc['decay_infty'])
    fig, axs = prepare_figure(
        ncols=len(rtypes_plot), figsize=(0.95, 0.36), wspace=0.20, bottom=0.29, left=0.15, 
        top=0.88, right=0.96)
    for jrtype, rtype in enumerate(rtypes_plot):
        axs[jrtype].axhline(0.0, c='#eeeeee', lw=0.5)
        for jdi, di in enumerate(decay_infty_plot):
            valid = np.ones(di_all.shape[0], dtype=np.bool8)
            for jd, d in enumerate(di):
                if d is not None: valid = np.logical_and(valid, di_all[:, jd] == d)
            y = 100 * (np.array(acc[rtype]) - np.array(acc['none']))/np.array(acc['none'])
            axs[jrtype].plot(
                di_all[valid, 1], y[valid], alpha=0.5, lw=1.0, c=colsbg[jdi], label=di[0])
            print(rtype, di[0], y[valid])
        axs[jrtype].text(
            0.03, 0.04, f'{ascii_lowercase[jrtype]})', transform=axs[jrtype].transAxes)
        axs[jrtype].set_ylim((-50, 2))
        axs[jrtype].set_xticks((0.0, 0.4, 0.8))
        axs[jrtype].text(
            0.50, -0.45, '$\\gamma_{\\infty}$ [-]', transform=axs[jrtype].transAxes, 
            ha='center')
        axs[jrtype].text(
            0.50, 1.04, labels[rtype], transform=axs[jrtype].transAxes, 
            ha='center', va='baseline', c='k')
    axs[0].text(
        -0.45, 0.50, 'accuracy RD [\%]', transform=axs[0].transAxes, rotation=90,
        ha='right', va='center')
    axs[2].legend(
        loc=7, bbox_to_anchor=(1.1, 0.45), frameon=False, title='$\\gamma_0$ [-]', 
        borderpad=0, labelspacing=0.38, handlelength=1.0, handletextpad=0.5, 
        borderaxespad=0.0)
#     plt.show()
    enforce_directory(os.path.dirname(fnfig))
    plt.savefig(fnfig)

if __name__ == '__main__':
    path0 = '/home/simon/Work/greg/parameters'
    L = 40
    params_rtypes = {rtype: read_parameters(L, pathp, rtype=rtype) for rtype in rtypes_def}

    coh_decay_t = (0.3, 0.6, 0.9)
    coh_infty_t = tuple(np.linspace(0.0, 0.95, num=20))
    
    fnsim = os.path.join(path0, 'assessment', 'coh.p')
    fnfig = os.path.join(path0, 'figures', 'assessment.pdf')
    acc = synthetic_scenario_coh(
            coh_decay_t, coh_infty_t, rtypes=rtypes_def, L=L, complex_reg=False, seed=1, 
            fnout=fnsim, overwrite=False)
    plot_synthetic(acc, fnfig)

    
