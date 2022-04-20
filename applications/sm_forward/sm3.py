from sm_forward import SMForward
import numpy as np
from matplotlib import pyplot as plt

# forward_model = SMForward(1/10, 0.02, 3)
# mirinov model

#


def fit_cubic(x, y):
    stack = np.stack([np.ones(x.shape[0]), x, np.sign(x) * np.abs(x)**(1/3)]).T
    fit = np.linalg.lstsq(stack, y)
    return fit[0]


def main():
    # degree = 2
    As = np.linspace(0.01, 0.1, 100)
    # As = np.array([0.007, 0.0072, 0.0075, 0.008])

    coeffs = np.zeros((As.shape[0], 2))
    for i in range(As.shape[0]):
        forward_model = SMForward(
            imag_slope=1/10, r_A=As[i], r_B=0.1, r_C=4)

        moistures = np.linspace(0, 50, 200)
        forward_model.set_moistures(moistures)
        print(As[i])
        forward_model.plot_dielectric()

        # sms = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

        n = 20000
        m1 = np.random.normal(loc=10, scale=2, size=n)
        m2 = np.random.normal(loc=10, scale=2, size=n)
        m3 = np.random.normal(loc=10, scale=2, size=n)

        idif12 = forward_model.dubois_I_dif(m1, m2)
        idif23 = forward_model.dubois_I_dif(m2, m3)
        idif13 = forward_model.dubois_I_dif(m3, m1)

        sm_dif = m2 - m1
        # plt.scatter(m2 - m1, idif12, s=5)
        # coeff, res, a, b, c = np.polyfit(
        #     sm_dif.flatten(), idif12.flatten(), deg=1, full=True)

        x1 = np.linspace((sm_dif.min()), sm_dif.max(), 100)
        # plt.plot(x1, np.polyval(coeff, x1))
        # plt.title(f'Slope: {coeff[0]}')
        # plt.xlabel('moisture difference')
        # plt.ylabel('Intensity difference')
        # plt.show()

        # plt.plot(m2 - m1, idif, '.')
        # plt.show()

        # plt.plot(idif, np.angle(phases), '.')
        # plt.xlabel('SM Difference')
        # plt.ylabel('Phase Difference')
        # plt.show()

        amp_triplet = (idif12 * idif23 * idif13)
        closure = forward_model.get_phases_dezan(
            m1, m2) * forward_model.get_phases_dezan(m2, m3) * forward_model.get_phases_dezan(m3, m1)

        coeff = fit_cubic((amp_triplet[amp_triplet != np.nan].flatten()), np.angle(
            closure[closure != np.nan]).flatten())

        coeff, res, a, b, c = np.polyfit((amp_triplet[amp_triplet != np.nan].flatten()), np.angle(
            closure[closure != np.nan]).flatten(), deg=1, full=True)
        # plt.scatter(amp_triplet, np.angle(closure), s=5)
        # x = np.linspace(amp_triplet.min(), amp_triplet.max(), 1000)
        # plt.plot(x, coeff[0] + coeff[1] * x + (coeff[2]
        #                                        * np.sign(x) * np.abs(x)**(1/3)))
        # plt.xlabel('Intensity Triplet')
        # plt.ylabel('Phase Triplet')
        # plt.show()

        # Plot density svatter plot with colors instead

        coeffs[i] = coeff

    # phis = []
    # for i in range(len(sms) - 1):
    #     phis.append(forward_model.get_phase_dezan(sms, 1, i))

    # plt.plot(np.angle(phis))
    # plt.xlabel('difference?')
    # plt.ylabel('Phase Difference')
    # plt.show()

    # plt.plot(sms, np.abs(sm_I))
    # plt.xlabel('SM')
    # plt.ylabel('Intensity')
    # plt.show()

    plt.scatter(As, coeffs[:, 1], s=5, label='intercept')
    plt.scatter(As, coeffs[:, 0], s=5, label='linear term')
    # plt.scatter(As, coeffs[:, 2], s=5, label='cubic root term')
    plt.legend(loc='upper right')
    plt.show()


main()
