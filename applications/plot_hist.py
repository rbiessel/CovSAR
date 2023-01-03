import logging
from cv2 import FONT_HERSHEY_SCRIPT_SIMPLEX
import numpy as np
import isceio as io
import os
from matplotlib import pyplot as plt
import figStyle
import seaborn as sns

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

stack1_folder = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures_dif_MLE'
stack2_folder = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures_dif_MLE'

hist_kwargs_b = {
    # 'bins': 100,
    # 'alpha': 0.65,
    # 'log_scale': (False, 10),
    'fill': True,
}


colors = ['#1f78b4', '#a6cee3']
colors = ['steelblue', 'tomato']

# colors = ['#001253', '#E14D2A']

filenames = ['correlation.fit', 'degree_0.fit',
             'degree_1.fit', 'cumulative_difference.fit']

titles = ['(a) Correlations',
          '(b) Slopes', '(c) Residual Means', '(d) End of Stack Displacement \nDifference']
# filenames = ['degree_0.fit']
fig, axes = plt.subplots(nrows=1, ncols=len(filenames), figsize=(12, 3))

for i in range(len(filenames)):

    filename = filenames[i]
    if len(filenames) == 1:
        ax = axes
    else:
        ax = axes[i]

    R2_v = io.load_file(os.path.join(stack1_folder, filename)).flatten()
    R2_d = io.load_file(os.path.join(stack2_folder, filename)).flatten()

    # if 'degree_0' in filename:
    #     R2_d = np.clip(R2_d, -17, 17)
    #     ax.set_xlim(-15, 15)

    # if 'cumulative_difference' in filename:
    #     R2_d = np.clip(R2_d, -7, 7)
    #     R2_v = np.clip(R2_v, -7, 7)
    #     ax.set_xlim(-5, 5)

    hist_kwargs = hist_kwargs_b.copy()

    if 'degree_0' in filename:
        # R2_v = np.sign(R2_v) * np.sqrt(np.abs(R2_v))
        # R2_d = np.sign(R2_d) * np.sqrt(np.abs(R2_d))
        # hist_kwargs['log_scale'] = 10
        hist_kwargs['clip'] = (-5, 5)
        x = np.linspace(-5, 5, 1000)
    elif 'correlation' in filename:
        x = np.linspace(0, 1, 1000)
    elif 'cumulative' in filename:
        x = np.linspace(-10, 10, 1000)
    elif 'degree_1' in filename:
        x = np.linspace(-1, 1, 1000)
    sns.kdeplot(R2_v,
                bw_adjust=.2, linewidth=3, alpha=0.5, color=colors[0], ax=ax, **hist_kwargs)
    sns.kdeplot(R2_d,
                bw_adjust=.2, linewidth=3, alpha=0.5, color=colors[1], ax=ax, **hist_kwargs)

    # ax.hist(R2_v, **hist_kwargs, label='Vegas', color=colors[0])
    # ax.hist(R2_d, **hist_kwargs, label='Dalton', color=colors[1])

    # ax.hist(R2_v, **hist_kwargs, color=colors[0], histtype='step')
    # ax.hist(R2_d, **hist_kwargs, color=colors[1], histtype='step')

    ax.set_title(titles[i], loc='left')


# Slope
axes[1].set_xlim([-5, 5])
axes[2].set_xlim([-1, 1])
axes[3].set_xlim([-5, 5])

axes[0].legend(loc='upper right')
axes[0].set_ylabel('Frequency')

axes[0].set_xlabel('Correlation Coefficient [$-$]')
axes[1].set_xlabel('m [$rad$]')
axes[2].set_xlabel('b [$rad$]')
axes[3].set_xlabel('$\Delta \Theta_{-1}$ [$mm$]')


plt.savefig('/Users/rbiessel/Documents/hist_fig.png', dpi=300)
plt.show()


# Plot Slope


# Plot
