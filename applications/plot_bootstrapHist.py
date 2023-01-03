from threading import main_thread
from matplotlib import pyplot as plt
import numpy as np
import figStyle
from greg.simulation import decay_model
import bootstrapCov


def plot_hist(correlations, r, rs_decays, decay):

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    bins = 50

    hist_kwargs = figStyle.hist_kwargs.copy()

    hist_kwargs['log'] = False

    ax[0].hist(correlations.flatten(), **hist_kwargs,
               color=figStyle.hist_colors[0], label='Speckle Realizations')

    ax[0].hist(correlations.flatten(),  **hist_kwargs,
               color=figStyle.hist_colors[0], histtype='step')

    ax[0].set_xlabel('Simulated R [$-$]')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('(a) Correlations from Observed Covariance', loc='left')
    ax[0].axvline(r, 0, 1, color='red',
                  label='Observation', alpha=0.8)

    lim = np.abs(r) + 0.1
    ax[0].set_xlim([-lim, lim])
    ax[0].legend(loc='best')

    ax[1].violinplot(
        rs_decays.T, positions=decay, widths=0.05, showmedians=True)
    ax[1].set_xlabel('Coherence Decay Rate')
    ax[1].set_ylabel('Correlation Coefficient')
    ax[1].set_title(
        '(b) Correlations from Various Coherence Decay Rates', loc='left')
    plt.tight_layout()
    plt.savefig('/Users/rbiessel/Documents/bootstrapHist.png', dpi=300)
    plt.show()
