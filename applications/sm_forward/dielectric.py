import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

mv = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
epsilon_r = np.array([2.3, 3.5, 5.5, 8, 11, 14.5, 18.5, 23, 28, 33, 39])
epsilon_r_i = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

mv2eps_real = interpolate.interp1d(mv, epsilon_r)
mv2eps_imag = interpolate.interp1d(mv, epsilon_r_i)

A, B, C = np.polyfit(mv, epsilon_r, 2)

print(A, B, C)
mv2 = np.linspace(0, 50, 100)


def mv2eps_real(sm):
    return C + B * sm + A * sm**2


# eps_real_fit =

# eps_real = mv2eps_real(mv2)
# eps_imag = mv2eps_imag(mv2)

plt.plot(mv2, mv2eps_real(mv2))
plt.plot(mv2, mv2eps_imag(mv2), '-')
plt.show()
