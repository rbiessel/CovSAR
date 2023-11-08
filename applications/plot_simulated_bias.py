import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import figStyle

fontsize = 13
colors = ['#647381', 'black']

scenarios = ['velDif', 'soilMoisture']

scenarioNames = ['Heterogenous Velocity', 'Dielectric Trend']

base_path = '/Users/rbiessel/Documents/igarss_paper/data/'


def plot_histograms():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    colors = ['tomato', 'steelblue']
    for i in range(len(scenarios)):
        closures = np.load(os.path.join(
            base_path, f'closures_{scenarios[i]}.npy'))
        sns.kdeplot(closures, ax=ax,
                    label=f'{scenarioNames[i]}', fill=True, color=colors[i], linewidth=2, bw_adjust=0.6)

    ax.legend(loc='upper left', fontsize=fontsize, framealpha=0)
    ax.set_xlabel('Closure Phase [$\mathrm{rad}$]', fontsize=fontsize)
    ax.set_ylabel('Density', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        '/Users/rbiessel/Documents/igarss_paper/figures/cphase_hist.png', dpi=300, transparent=True)
    plt.show()


# def plot_histograms():
#     fig, axes = plt.subplots(nrows=1, ncols=len(scenarios))
#     for i in range(len(scenarios)):
#         ax = axes[i]
#         closures = np.load(os.path.join(
#             base_path, f'closures_{scenarios[i]}.npy'))
#         sns.kdeplot(closures, ax=ax)
#         ax.set_title(scenarioNames[i])
#         ax.set_xlabel('Closure Phase [$rad$]')
#         ax.set_ylabel('Density')


def plot_rawerror():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    scatter_styles = ['x', 'o']
    baselines = np.load(os.path.join(
        base_path, f'baselines.npy'))
    alphas = [1, 0.5]
    for i in range(len(scenarios)):
        errors = np.load(os.path.join(
            base_path, f'phierrors_{scenarios[i]}.npy'))

        errors = errors * 56 / (np.pi * 4)

        ax.scatter(baselines[baselines > 0], errors[baselines > 0], 30, marker=scatter_styles[i], alpha=alphas[i],
                   label=f'{scenarioNames[i]}', color=colors[i])
        ax.set_ylabel('Deformation Error [$\mathrm{mm}$]', fontsize=fontsize)
        ax.set_xlabel('Temporal Baseline [$\mathrm{days}$]', fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize, framealpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(
        '/Users/rbiessel/Documents/igarss_paper/figures/phierror.png', dpi=300, transparent=True)
    plt.show()


def plot_maxb():
    fontsize = 13
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    maxb = np.load(os.path.join(
        base_path, f'maxbaseline.npy'))

    linestyles = ['-', '--']

    ax = axes
    for i in range(len(scenarios)):
        error_high = np.load(os.path.join(
            base_path, f'error_{scenarios[i]}_high.npy'))

        ax.plot(maxb * 12, error_high, linestyles[i],
                label=f'{scenarioNames[i]}', color=colors[i], linewidth=4)

    ax.legend(loc='best', fontsize=fontsize)
    ax.set_xlabel(
        'PL Maximum Temporal Baseline [$\mathrm{days}$]', fontsize=fontsize)
    ax.set_ylabel('Bias [$\mathrm{mm}/\mathrm{yr}$]', fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()

    plt.savefig(
        '/Users/rbiessel/Documents/igarss_paper/figures/maxb.png', dpi=300, transparent=True)
    plt.show()


plot_histograms()
plot_rawerror()
plot_maxb()
