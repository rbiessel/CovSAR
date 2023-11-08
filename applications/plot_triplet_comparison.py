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

stack_folder = '/Users/rbiessel/Documents/InSAR/vegas_all/DNWR/'

forms = ['triple', 'arctan', 'tanh', 'logistic']
labels = ['Triple Product', 'arctan', 'tanh', 'Logistic']


hist_kwargs_b = {
    # 'bins': 100,
    # 'alpha': 0.65,
    # 'log_scale': (False, 10),
    'fill': False,
}


colors = ['#1f78b4', '#a6cee3']
colors = ['steelblue', 'tomato']

filename = 'correlation.fit'


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

for i in range(len(forms)):

    path = os.path.join(stack_folder, forms[i], filename)

    data = io.load_file(path).flatten()**2

    hist_kwargs = hist_kwargs_b.copy()

    sns.kdeplot(data,
                bw_adjust=.2, linewidth=2, alpha=0.8, label=labels[i], ax=ax, fill=False, clip=(0, 1), cut=0)


ax.legend(loc='best')
ax.set_xlabel('$R^{2}$')
ax.set_ylabel('pdf')

plt.savefig(
    '/Users/rbiessel/Documents/InSAR/closure_manuscript/figures/triplet_comparison.png', dpi=300)
plt.show()
