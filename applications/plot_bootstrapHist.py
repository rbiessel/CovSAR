from pub_pixels import pixel_paths
from threading import main_thread
from matplotlib import pyplot as plt
import numpy as np
import figStyle
from greg.simulation import decay_model
import bootstrapCov
import seaborn as sns
import os

colors = ['steelblue', 'tomato', '#646484']


data = np.array(pixel_paths).reshape((2, 3))


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))

print(axes.shape)
print(data.shape)

hist_kwargs = figStyle.hist_kwargs.copy()

labels = np.array([['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']])

print(labels.shape)
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        ax = axes[i, j]

        hist_kwargs['log'] = False

        r_sim = np.load(os.path.join(data[i, j], 'rs_sim.npy'))
        R_sigma_p = np.load(os.path.join(data[i, j], 'R_sigma_p.npy'))
        r = R_sigma_p[0]

        label = data[i, j].split('/')[-3]
        sns.kdeplot(r_sim.flatten(),
                    bw_adjust=.5, color=colors[2], ax=ax, fill=True, alpha=0.1, linewidth=2, clip=(-1, 1))

        ax.set_xlabel('R [$-$]')
        ax.set_ylabel('')
        # ax[i].set_title(label)
        ax.axvline(r, 0, 1, color=colors[1],
                   label='Observation', alpha=1)

        lim = np.max(r_sim) + 0.1
        if np.abs(r) > np.max(r_sim):
            lim = np.abs(r) + 0.1

        ax.set_xlim([-lim, lim])
        yticks = [0, 1, 2]
        ax.set_yticks(yticks, labels=yticks)
        ax.set_title(labels[i, j], loc='left')
    axes[i, 0].set_ylabel('Density')

plt.tight_layout()
plt.savefig(
    '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/bootstrapHist.png', dpi=300)
plt.show()
