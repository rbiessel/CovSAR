'''
Created on Feb 10, 2022

@author: simon
'''
import os
import numpy as np

from greg import load_object, save_object, paramorder, read_parameters
from scipy.special import expit

paramnames = {
    'hadamard': 'hadreglparam', 'hadspec': 'hadspecreglparam', 'spectral': 'specregparam'}

def _read_looks(path1):
    def _l_from_fn(fn):
        try:
            l = int(os.path.splitext(fn)[0])
        except:
            l = None
        return l
    looks = [_l_from_fn(x) for x in os.listdir(path1)]
    return sorted([l for l in looks if l is not None])

def read_params_raw(path0, rtype='hadamard', rmatrix='G', scenario='broad', verbose=False):
    path1 = os.path.join(path0, rtype, rmatrix, scenario)
    looks = _read_looks(path1)
    params = []
    for l in looks:
        res = load_object(os.path.join(path1, f'{l}.p'))
        p = res[paramnames[rtype]]
        if verbose:
            print(l, p, res['f'], res['f_noreg'], res['f'] / res['f_noreg'])
        params.append(p)
    params = np.array(params)
    return params, looks

def transform_looks(looks):
#     tlooks = np.sqrt(np.array(looks))
    tlooks = np.array(looks)
    return tlooks

def fit_params_looks(params, looks, looks_d=None):
    Np = np.shape(params)[1]
    tlooks = transform_looks(looks)
    params_d = []
    if looks_d is None: looks_d = np.arange(16, 401)
    def _fit(tlooks, p, tlooks_d):
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        predicted = lowess(p, tlooks, xvals=tlooks_d, frac=0.50, it=1, is_sorted=True)
        return expit(predicted)
    for jp in range(Np):
        params_ = params[:, jp]
        predicted = _fit(tlooks, params_, transform_looks(looks_d))
        params_d.append(predicted)
    return looks_d, np.array(params_d)

def export_parameters(path0):
    path_out = os.path.join(path0, 'parameters')
    looks_d = np.arange(16, 401)
    for rtype in paramnames:
        params, looks = read_params_raw(path0, rtype)
        _, params_d = fit_params_looks(params, looks, looks_d=looks_d)
        paramdict = {'params': params_d, 'looks': looks_d, 'names': paramorder[rtype]}
        save_object(paramdict, os.path.join(path_out, f'{rtype}.p'))

def plot_parameter_fit(path0):
    rtypes = ['hadamard', 'spectral', 'hadspec']
    labels = {'beta': '$\\beta$', 'alpha': '$\\alpha$', 'nu': '$\\nu$'}
    pos = {'hadamard': {'alpha': (1.00, 0.81), 'nu': (1.00, 0.90)},
           'hadspec': {'alpha': (0.10, 0.74), 'nu': (0.14, 0.40), 'beta': (1.00, 0.03)},
           'spectral': {'beta': (1.00, 0.47)}}
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, colsbg
    fig, axs = prepare_figure(
        ncols=len(rtypes), figsize=(1.95, 0.4), wspace=0.30, bottom=0.24, left=0.07,
        top=0.97, right=0.96)
    for jrt, rt in enumerate(rtypes):
        paramdict = load_object(os.path.join(path0, 'parameters', f'{rt}.p'))
        paramsraw, looks = read_params_raw(path0, rt)
        paramnames = paramorder[rt]
        assert paramnames == paramdict['names']
        print(paramnames)
        for jpn, pn in enumerate(paramnames):
            axs[jrt].plot(
                looks, expit(paramsraw[:, jpn]), linestyle='none', marker='o', markersize=3,
                c=colsbg[jpn], mec='none', alpha=0.5, zorder=1)
            axs[jrt].plot(
                paramdict['looks'], paramdict['params'][jpn], lw=0.5, c=colsbg[jpn],
                zorder=2, alpha=0.5)
            axs[jrt].text(*pos[rt][pn], labels[pn], transform=axs[jrt].transAxes)
        axs[jrt].text(
            0.5, -0.3, 'looks $L$ [-]', ha='center', va='baseline', 
            transform=axs[jrt].transAxes)
    axs[0].text(
        -0.20, 0.50, 'parameter [-]', ha='right', va='center', rotation=90,
        transform=axs[0].transAxes)
    plt.savefig(os.path.join(path0, 'figures', 'parameters.pdf'))



if __name__ == '__main__':
    path0 = '/home/simon/Work/greg'
#     export_parameters(path0)
    print(read_parameters(65, os.path.join(path0, 'parameters'), rtype='hadspec'))
#     plot_parameter_fit(path0)
