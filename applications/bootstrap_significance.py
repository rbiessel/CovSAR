from pub_pixels import pixel_paths
import statsmodels.api as sm
from scipy.special import ndtr
import scipy as scipy
import logging
import cphase_pred
import os
import closures
from covariance import CovarianceMatrix
from bootstrapCov import bootstrap_correlation
from greg import simulation as greg_sim
from matplotlib import pyplot as plt
import numpy as np

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


# data_root = '/Users/rbiessel/Documents/InSAR/plotData/imnav/p_75_62'
# data_root = '/Users/rbiessel/Documents/InSAR/plotData/imnav/p_77_58'
# data_root = '/Users/rbiessel/Documents/InSAR/plotData/DNWR/p_10_100'


for data_root in pixel_paths:

    C = np.load(os.path.join(
                data_root, 'C_raw.np.npy'))

    intensity_triplets = np.load(os.path.join(data_root, 'ampTriplets.np.npy'))
    phase_triplets = np.load(os.path.join(data_root, 'closures.np.npy'))
    intensities = np.load(os.path.join(data_root, 'Intensities.np.npy'))

    lf = 14
    ml_size = (1*lf, 7*lf)
    triplets = closures.get_triplets(C.shape[0], all=False)
    A, rank = closures.build_A(triplets, C)
    print(A)
    Adag = np.linalg.pinv(A)

    error_coh, R, coeff = cphase_pred.regress_intensity(
        phase_triplets, intensity_triplets, intensities, A, Adag, C.shape[0], plot=False)

    l = int(
        np.floor(np.sqrt(ml_size[0] * ml_size[1]))/2)
    samples = 10000

    plt.imshow(np.abs(C))
    plt.show()

    rs_sim = bootstrap_correlation(
        C, l, triplets, nsample=samples, fitLine=False, zeroPhi=True)

    print('Mean: ', np.mean(rs_sim))
    print('STD: ', np.std(rs_sim))
    std = np.std(rs_sim)
    print('Number of Sigmas: ', np.abs(R)/np.std(rs_sim))

    ecdf = sm.distributions.ECDF(rs_sim)

    plt.hist(rs_sim, bins=100)
    plt.show()

    to_save = np.array([R, np.abs(R)/np.std(rs_sim), ecdf(-1 * np.abs(R))])
    np.save(os.path.join(data_root, 'rs_sim'), rs_sim)
    np.save(os.path.join(data_root, 'R_sigma_p'), to_save)

# plot_hist(rs_sim, R, rs_decays, decay)
