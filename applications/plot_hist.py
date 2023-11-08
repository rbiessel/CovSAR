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

stack1_folder = '/Users/rbiessel/Documents/InSAR/vegas_all/subsetA/closures'
stack2_folder = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/dalton/closures'

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
             'degree_1.fit', 'max_difference.fit', 'bias.fit']

titles = ["(a) Correlation",
          '(b) Slopes', '(c) Residual Mean Systematic Closure Phase', '(d) Largest Magnitude Displacement Difference', '(e) Displacement Rate Bias']

labels = ['correlation', 'slopes', 'intercepts', 'max', 'bias']

for i in range(len(filenames)):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    filename = filenames[i]
    ax = axes

    R2_v = io.load_file(os.path.join(stack1_folder, filename)).flatten()
    clip_vegas = [0, -1, 0, -1]

    R2_d = io.load_file(os.path.join(stack2_folder, filename)).flatten()
    clip_dalton = [0, -1, 0, -1]

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
        hist_kwargs['clip'] = (-1.5, 1.5)
        x = np.linspace(-1.5, 1.5, 1000)

        print('Slope Statistics:')
        print(f'Vegas Median: {np.median(R2_v)}')
        print(f'Vegas STD: {np.std(R2_v)}')
        ax.set_xlabel('m [$rad$]')
        print(f'Dalton Median: {np.median(R2_d)}')
        print(f'Dalton STD: {np.std(R2_d)}')
        print('\n')
        ax.set_xlim([-1.5, 1.5])

    elif 'correlation' in filename:
        x = np.linspace(-1, 1, 1000)
        print('R^2 Statistics:')
        print(f'Vegas Median: {np.median(R2_v**2)}')
        print(f'Vegas STD: {np.std(R2_v**2)}')

        print(f'Dalton Median: {np.median(R2_d**2)}')
        print(f'Dalton STD: {np.std(R2_d**2)}')
        print('\n')
        ax.set_xlabel('Correlation Coefficient [$-$]')

    elif 'bias' in filename:
        R2_v *= 365/12
        R2_d *= 365/12
        hist_kwargs['clip'] = (-1, 1)
        x = np.linspace(-1, 1, 1000)
        print('Bias Statistics:')
        print(f'Vegas Median: {np.median(np.abs(R2_v))}')
        print(f'Vegas STD: {np.std(np.abs(R2_v))}')

        print(f'Dalton Median: {np.median(np.abs(R2_d))}')
        print(f'Dalton STD: {np.std(np.abs(R2_d))}')
        print('\n')
        ax.set_xlabel('$\Theta Bias$ [mm/month]')
        ax.set_xlim([-0.5, 0.5])

    elif 'max_difference' in filename:
        hist_kwargs['clip'] = (-5, 5)
        x = np.linspace(-5, 5, 1000)
        print('Max Difference Statistics:')
        print(f'Vegas Median: {np.median(np.abs(R2_v))}')
        print(f'Vegas STD: {np.std(np.abs(R2_v))}')
        print(f'Dalton Median: {np.median(np.abs(R2_d))}')
        print(f'Dalton STD: {np.std(np.abs(R2_d))}')
        print('\n')
        ax.set_xlim([-5, 5])
        ax.set_xlabel(r'$\Delta \theta_{\mathrm{smax}}$ [mm]')

    elif 'degree_1' in filename:
        hist_kwargs['clip'] = (-0.4, 0.75)
        x = np.linspace(-0.4, 0.75, 1000)
        print('Intercept Statistics:')
        print(f'Vegas Median: {np.median(R2_v)}')
        print(f'Vegas STD: {np.std(R2_v)}')

        print(f'Dalton Median: {np.median(R2_d)}')
        print(f'Dalton STD: {np.std(R2_d)}')
        print('\n')
        ax.set_xlim([-0.4, 0.75])
        ax.set_xlabel('b [$rad$]')

    sns.kdeplot(R2_d,
                bw_adjust=.2, linewidth=1.5, alpha=0.5, label='Dalton Highway', color=colors[1], ax=ax, **hist_kwargs)
    sns.kdeplot(R2_v,
                bw_adjust=.2, linewidth=1.5, alpha=0.5, linestyle='dashed', label='Las Vegas', color=colors[0], ax=ax, **hist_kwargs)

    if 'bias' in filename:
        ax.legend(loc='upper right', fontsize=14)

    # ax.hist(R2_v, **hist_kwargs, label='Vegas', color=colors[0])
    # ax.hist(R2_d, **hist_kwargs, label='Dalton', color=colors[1])

    # ax.hist(R2_v, **hist_kwargs, color=colors[0], histtype='step')
    # ax.hist(R2_d, **hist_kwargs, color=colors[1], histtype='step')

    ax.set_title(titles[i], loc='left', fontsize=14)

    # Slope
    # intercept
    # displacement

    # axes[0].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(
        f'/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/hist_{labels[i]}.png', dpi=300)
    plt.show()