from cv2 import mean
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
import closures
from sm_forward.sm_forward import SMForward
import library as sarlab
from scipy.stats import gaussian_kde


dezan = SMForward(imag_slope=0.1, r_A=0.05, r_B=0.1, r_C=4)


def build_nl_coherence(nl_func, sm):
    '''
      Given a time-series of sm, compute a simulated coherence matrix with systematic phase closures
    '''
    coherence = np.zeros((sm.shape[0], sm.shape[0]), dtype=np.cdouble)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = nl_func(sm[i] - sm[j])  # * nl_func(sm[j]).conj()

    return coherence


def sm_to_phase_sqrt(sm):
    '''
      Convert an array of volumetric soil moistures to angles according to a square root function.
      Soil moistures betwen 0 and 50 will return angles in the range [0, pi/2] in the form of a complex array
    '''
    sign = np.sign(sm)
    angle = (np.sqrt(2) * np.pi)**-1 * np.sqrt(np.abs(sm))
    return np.exp(1j * sign * angle)


def build_nl_coherence_dezan(sm, dezan):
    coherence = np.ones((sm.shape[0], sm.shape[0]), dtype=np.complex64)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[0]):
            coherence[i, j] = dezan.get_phases_dezan(sm[i], sm[j])

    return coherence


def multilook(image, looks):
    image = uniform_filter(image.real, size=looks) + \
        1j * uniform_filter(image.imag, size=looks)

    return image


def get_velocity(scaling_factor=5, base_velocity=1):
    '''
        Scaling factor defines how many times faster the

    '''
    # Load root image
    path = './polygon_drawing.jpg'
    col = Image.open(path)
    gray = col.convert('L')
    gray = np.array(gray)
    filtered = uniform_filter(gray, size=50)[::30, ::30][10:20, 10:20]
    amplitude = filtered / 255
    velocity = 1 - amplitude

    velocity += base_velocity
    velocity *= scaling_factor

    return velocity, amplitude


def main():

    # velocity, amplitude = get_velocity(base_velocity=1, scaling_factor=5)
    velocity, amplitude = get_velocity(base_velocity=0, scaling_factor=0)

    shape = velocity.shape
    print(f'Shape: {shape}')

    velocity_noise = np.random.normal(
        loc=0, scale=0, size=(shape[0] * shape[1])).reshape(shape)

    velocity += velocity_noise
    velocity = np.abs(velocity)

    # velocity m/yr  > cm / day

    velocity = -1 * velocity / 365

    plt.imshow(velocity * 365)
    plt.colorbar(label='cm/year')
    plt.show()

    lookvector = np.linspace(1, 2000, 20)
    mean_closures = np.zeros(lookvector.shape)
    max_closures = np.zeros(lookvector.shape)
    # amplitude = 1

    n = 20
    days = np.arange(0, 12*n, 12)
    baslines = np.tile(days, (len(days), 1))
    baslines = baslines.T - baslines

    # np.random.seed(10)

    # sm_stack = np.zeros((n, amplitude.shape[0], amplitude.shape[1]))
    # sm = np.abs(np.random.normal(scale=10, loc=25, size=n))
    sm = np.linspace(40, 10, n) + np.random.normal(scale=1, loc=0, size=n)
    # size = sm_stack.shape[1] * sm_stack.shape[2]
    # for i in range(n):
    #     base_sm = sm[i]
    #     sm_stack[i] = ((1 - amplitude) + 0.5) * np.abs(np.random.normal(
    #         loc=base_sm, scale=1, size=size)).reshape((sm_stack.shape[1], sm_stack.shape[2]))

    # sm_stack[i] = np.abs(np.random.normal(
    #     loc=base_sm, scale=3, size=size)).reshape((sm_stack.shape[1], sm_stack.shape[2]))
    # sm[i] = np.mean(sm_stack[i])

    # plt.imshow(sm_stack[i], vmin=0, vmax=100)
    # plt.show()

    # sm = np.linspace(50, 10, n) + np.random.normal(scale=4, loc=0, size=n)
    # sm = np.ones(n)
    # dezan.set_moistures(sm)
    # dezan.plot_dielectric()

    # plt.hist(sm, bins=25)
    # plt.show()

    coherence = np.ones((n, n), dtype=np.complex64)
    true_coherence = np.ones((n, n), dtype=np.complex64)

    # sm_coherence_stack = np.ones(
    #     (n, n, sm_stack.shape[1], sm_stack.shape[2]), dtype=np.complex64)

    # print('Computing Soil Moisture Phases')
    # for k in range(sm_coherence_stack.shape[2]):
    #     for l in range(sm_coherence_stack.shape[3]):
    #         print(k, l)
    #         sm_coherence_stack[:, :, k, l] = build_nl_coherence_dezan(
    #             sm_stack[:, k, l], dezan)

    print('Computing Observed Phase')

    for i in range(n):
        for j in range(n):
            phi = np.exp(
                1j * ((velocity * (days[j] - days[i])) * 4 * np.pi / 5.6))

            # print(sm_coherence_stack[j, i].shape)
            # phi = phi * sm_coherence_stack[i, j]
            # phi = phi * build_nl_coherence(sm_to_phase_sqrt, sm)

            # plt.imshow(np.abs(phi), vmin=0, vmax=1)
            # plt.show()

            coherence[i, j] = np.mean(phi.real) + 1j * np.mean(phi.imag)
            true_coherence[i, j] = np.exp(
                1j * (np.mean((velocity) * (days[j] - days[i])) * 4 * np.pi / 5.6))

    # coherence = coherence * build_nl_coherence(sm_to_phase_sqrt, sm)
    # coherence = coherence * build_nl_coherence(sm_to_phase_sqrt, sm)
    coherence = coherence * build_nl_coherence_dezan(sm, dezan)

    triplets = closures.get_triplets(n, all=False)

    print(triplets)
    # triplets = np.array([triplet for triplet in triplets if triplet[0] == 0])

    triplets_permuted_1 = [[triplet[0], triplet[2], triplet[1]]
                           for triplet in triplets]

    # triplets = np.concatenate(
    #     (triplets, triplets_permuted_1))

    A, rank = closures.build_A(triplets, coherence)
    U, S, Vh = np.linalg.svd(A)
    A_dagger = Vh[:rank].T @ np.diag(1/S[:rank]) @ U.T[:rank]

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].set_title('Observed Phases')
    im = ax[0].imshow(np.angle(coherence), vmin=-np.pi,
                      vmax=np.pi, cmap=plt.cm.seismic)
    ax[1].set_title('True Phases')
    ax[1].imshow(np.angle(true_coherence), vmin=-np.pi,
                 vmax=np.pi, cmap=plt.cm.seismic)
    ax[2].set_title('Normalized Difference')
    ax[2].imshow(np.angle(coherence * true_coherence.conj()) / np.angle(coherence * true_coherence), vmin=-np.pi,
                 vmax=np.pi, cmap=plt.cm.seismic)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax,  label='Radians')
    plt.show()

    plt.scatter((baslines).flatten(), (np.angle(
        coherence * true_coherence.conj())).flatten(), s=10)
    plt.xlabel('Temporal Basline')
    plt.ylabel('Phase Error')
    plt.show()

    sim_closures = np.zeros(len(triplets), dtype=np.complex64)
    amp_triplets = np.zeros(len(triplets))

    for i in range(len(triplets)):
        triplet = triplets[i]
        closure = coherence[triplet[0], triplet[1]] * coherence[triplet[1],
                                                                triplet[2]] * coherence[triplet[0], triplet[2]].conj()

        amp_triplet = sarlab.intensity_closure(
            sm[triplet[0]], sm[triplet[1]], sm[triplet[2]], norm=False, cubic=False, legacy=False)

        sim_closures[i] = closure
        amp_triplets[i] = amp_triplet

    # Check model consistency

    closure_4 = np.angle(
        sim_closures[0]) - np.angle(sim_closures[1]) + np.angle(sim_closures[2])

    amp_closure4 = amp_triplets[0] - amp_triplets[1] + amp_triplets[2]

    print('Testing model consistency: ')
    print(np.abs((closure_4) - np.angle(sim_closures[3]))**2)
    print(np.abs(amp_closure4 - amp_triplets[3])**2)

    form = 'linear'
    coeff, covm = sarlab.gen_lstq(
        amp_triplets, np.angle(sim_closures), function='lineari')

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # print(triplets.shape)
    # print(sm[triplets[:, 0]])

    x = 1 * (sm[triplets[:, 1]] - sm[triplets[:, 0]])
    y = 1 * (sm[triplets[:, 2]] - sm[triplets[:, 1]])
    z = 1 * (sm[triplets[:, 0]] - sm[triplets[:, 2]])
    c = closures.eval_sytstematic_closure(
        amp_triplets.flatten(), coeff, form='linear')
    c2 = np.angle(sim_closures)

    maxc = np.abs(np.max(c))

    # img = ax.scatter(x, y, z, c=c, cmap=plt.cm.seismic, vmin=-maxc, vmax=maxc)
    img = ax.scatter(x, y, z, c=c2, cmap=plt.cm.seismic, vmin=-maxc, vmax=maxc)

    fig.colorbar(img)
    plt.show()

    x = np.linspace(amp_triplets.min(), amp_triplets.max(), 100)

    xy = np.vstack([amp_triplets.flatten(), np.angle(sim_closures).flatten()])
    z = gaussian_kde(xy)(xy)

    plt.scatter(amp_triplets, np.angle(sim_closures), c=z, s=10)
    plt.title(
        f'Slope: {np.format_float_scientific(coeff[0], precision=3)} Intercept: {np.format_float_scientific(coeff[1], precision=3)}')
    plt.ylabel('Closure Phase (Radians)')
    plt.xlabel('Logistic Intensity Triplet')
    plt.plot(x, closures.eval_sytstematic_closure(
        x, coeff, form='lineari'), '--', label='With Intercept')
    plt.plot(x, closures.eval_sytstematic_closure(
        x, coeff, form='linear'), '--', label='Without Intercept')
    plt.grid(alpha=0.5)
    plt.legend(loc='lower right')
    plt.show()

    plt.hist(np.angle(sim_closures), bins=50)
    plt.xlabel('Closure Phase (Radians)')
    plt.ylabel('Count')
    plt.show()

    systematic_phi_errors = closures.least_norm(
        A, np.angle(sim_closures), pinv=False, pseudo_inv=A_dagger)

    error_coh = closures.phivec_to_coherence(systematic_phi_errors, n)

    plt.title('Phases inverted from closure phases')
    plt.imshow(np.angle(error_coh), vmin=-np.pi/4,
               vmax=np.pi/4, cmap=plt.cm.seismic)
    plt.show()

    return

    for i in range(lookvector.shape[0]):

        phi_12 = (velocity * 36) * 4 * np.pi / 5.6

        phi_23 = (velocity * 24) * 4 * np.pi / 5.6

        phi_13 = (velocity * 60) * 4 * np.pi / 5.6

        phi_12 = amplitude * np.exp(1j * phi_12)
        phi_23 = amplitude * np.exp(1j * phi_23)

        phi_13 = amplitude * np.exp(1j * phi_13)

        looks = (lookvector[i], lookvector[i])
        phi_12 = multilook(phi_12, looks)
        phi_23 = multilook(phi_23, looks)

        phi_13 = multilook(phi_13, looks)

        closure = np.angle(phi_12 * phi_23 * phi_13.conj())
        # plt.imshow(closure)
        # plt.show()

        max_closures[i] = np.max(closure)
        mean_closures[i] = np.mean(closure)

    # plt.imshow(np.angle(image))
    # plt.show()

    # plt.imshow(np.angle(phi_12 * phi_12 * phi_13.conj()))
    # plt.show()

    plt.scatter(lookvector**2, mean_closures, s=10, label='Mean Closures')
    plt.scatter(lookvector**2, max_closures, s=10, label='Max Closures')
    plt.legend(loc='lower right')
    plt.ylabel('Closure (Radians)')
    plt.xlabel('looks')
    plt.show()


main()
