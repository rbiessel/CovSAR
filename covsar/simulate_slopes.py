import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
import library as sarlab
from covariance import CovarianceMatrix


def main():

    n = 4
    m = 100

    for cvi in np.linspace(0, 2, 10):
        print(cvi)
        cv = np.array([[1, cvi], [cvi, 1]])

        samples = np.random.multivariate_normal(
            [0, 0], cv, size=m*m*n, check_valid='warn', tol=1e-8)

        samples = samples.reshape((n, m, m, 2))
        stack = samples[:, :, :, 0] + samples[:, :, :, 1] * 1j
        # stack.imag = np.sign(stack.real) * (stack.real**2) * 1j
        plt.scatter(stack.real, stack.imag)
        plt.show()

        cov = CovarianceMatrix(stack, ml_size=(41, 41))
        coherence = cov.get_coherence()
        intensities = cov.get_intensity()
        print(intensities.shape)

        closure = coherence[0, 1] * coherence[1, 2] * coherence[2, 0]
        amp_triplet = sarlab.intensity_closure(
            intensities[:, :, 0], intensities[:, :, 1], intensities[:, :, 2])

        plt.imshow(np.angle(closure))
        plt.show()

        plt.imshow(amp_triplet)
        plt.show()

        sampling = (np.abs(np.random.rand(5, 5)) *
                    amp_triplet.shape[0]).astype(np.int8)
        plt.scatter(amp_triplet[sampling].flatten(),
                    np.angle(closure)[sampling].flatten(), s=10)
        plt.show()


main()
