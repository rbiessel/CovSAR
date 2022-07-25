import matplotlib.pyplot as plt
globfigparams = {
    'fontsize':8, 'family':'serif', 'usetex': True, 
    'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
    'column_inch':229.8775 / 72.27, 'markersize':24, 'markercolour':'#AA00AA',
    'fontcolour':'#666666', 'tickdirection':'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1 }

colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']

def prepare_figure(
        nrows=1, ncols=1, figsize=(1.7, 0.8), figsizeunit='col', sharex='col', sharey='row',
        squeeze=True, bottom=0.10, left=0.15, right=0.95, top=0.95, hspace=0.5, wspace=0.1, 
        remove_spines=True, gridspec_kw=None, subplot_kw=None):
    plt.rc('font', **{'size':globfigparams['fontsize'], 'family':globfigparams['family']})
    plt.rcParams['text.usetex'] = globfigparams['usetex']
    plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
    plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
    plt.rcParams['font.size'] = globfigparams['fontsize']
    plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
    plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
    plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
    plt.rcParams['xtick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
    plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
    plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
    plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
    plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']

    plt.rcParams['text.color'] = globfigparams['fontcolour']
    if figsizeunit == 'col':
        width = globfigparams['column_inch']
    elif figsizeunit == 'in':
        width = 1.0

    figprops = dict(facecolor='white', figsize=(figsize[0] * width, figsize[1] * width))
    if nrows > 0 and ncols > 0:
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols , sharex=sharex, sharey=sharey, squeeze=squeeze,
            gridspec_kw=gridspec_kw, subplot_kw=subplot_kw)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top, hspace=hspace,
                            wspace=wspace)
    else:
        fig = plt.figure()
        axs = None
    fig.set_facecolor(figprops['facecolor'])
    fig.set_size_inches(figprops['figsize'], forward=True)
    if remove_spines:
        try:
            for ax in axs.flatten():
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        except:
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
    return fig, axs
