import numpy as np
from matplotlib import pyplot as plt
from greg import simulation as greg_sim
from bootstrapCov import bootstrap_correlation
from covariance import CovarianceMatrix
import closures
import os
from plot_bootstrapHist import plot_hist

data_root = '/Users/rbiessel/Documents/InSAR/plotData/imnav/p_75_62'

C = np.load(os.path.join(
            data_root, 'C_raw.np.npy'))
lf = 4
ml_size = (7*lf, 19*lf)
triplets = closures.get_triplets(C.shape[0], all=False)

print(C, triplets)


l = int(
    np.floor(np.sqrt(ml_size[0] * ml_size[1]))/2)

decay = np.array(
    [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
samples = 1000
rs_decays = np.zeros((decay.shape[0], samples))
for i in range(len(decay)):
    C_decay = greg_sim.decay_model(
        R=1, L=l, P=C.shape[0], coh_decay=decay[i], coh_infty=0.05, returnC=True)
    print(C_decay.shape)
    rs_decay = bootstrap_correlation(
        C_decay, l, triplets, nsample=samples, fitLine=False)
    rs_decays[i] = rs_decay

rs_sim, coeffs_sim = bootstrap_correlation(
    C, l, triplets, nsample=1000, fitLine=True, zeroPhi=True)

plot_hist(rs_sim, r, rs_decays, decay)
fig, ax = plt.subplots(ncols=3, nrows=1)
bins = 100
ax[0].hist(rs_sim.flatten(), bins=bins)
ax[0].set_title('Rs')

ax[1].hist(coeffs_sim[0].flatten(), bins=bins)
ax[1].set_title('Slope')

ax[2].hist(coeffs_sim[1].flatten(), bins=bins)
ax[2].axvline(coeff[1], 0, 1, color='red')
ax[2].set_title('Mean Residual Phase')

plt.show()
