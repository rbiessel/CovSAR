from matplotlib import pyplot as plt
globfigparams = {
    'fontsize': 8, 'family': 'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{times} \usepackage{mathtools}',
    'column_inch': 229.8775 / 72.27, 'markersize': 24, 'markercolour': '#AA00AA',
    'fontcolour': 'black', 'tickdirection': 'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1}

# colsbg = ['#19192a', '#626282', '#aaaabe', '#cbcbd7']
plt.rc('font', **
       {'size': globfigparams['fontsize']})
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

hist_kwargs = {
    'bins': 100,
    'alpha': 0.65,
    'log': True,
    'linewidth': 2,
}

hist_colors = ['#1f78b4', '#a6cee3']
