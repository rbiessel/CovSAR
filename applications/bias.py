import numpy as np

from matplotlib import pyplot as plt


def main():

    t = np.linspace(0, 120, 240)

    velocity_1 = 1  # cm/d
    ps_bias = 1  # cm/d
    ds_bias = ps_bias * 2

    coherence = np.zeros()

    displacement_true = t * velocity_1
    displacement_ps = t * (velocity_1 + ps_bias)
    displacement_ds = t * (velocity_1 + ds_bias)

    plt.plot(t, displacement_true, '--', label='Displacement True')
    plt.plot(t, displacement_ps, '--', label='Displacement PS')
    plt.plot(t, displacement_ds, '--', label='Displacement DS')
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.show()


main()
