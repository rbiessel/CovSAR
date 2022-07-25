'''
Created on Nov 17, 2021

@author: szwieback
'''

import numpy as np
np.set_printoptions(precision=2, suppress=True)

M1 = 4
M2 = 4

b = 0.9
rho = 0.8
u1 = np.zeros(M1)
u2 = np.zeros(M2)
u1[1:3] = 1/np.sqrt(2)
u2[0:3] = 1/np.sqrt(3)

Sigma = np.eye(M1 + M2)
Sigma[:M1, M1:] = b * np.outer(u1, u2.conj())
Sigma[M1:,:M1] = Sigma[:M1, M1:].T.conj()

C = np.eye(M1 + M2)
C[:M1, M1:] = rho * np.outer(u1, u2.conj())
C[M1:,:M1] = C[:M1, M1:].T.conj()

Sigmainv = (1 / (1 - b ** 2)) * np.eye(M1 + M2)
Sigmainv[:M1,:M1] += (b ** 2 / (1 - b ** 2)) * (np.outer(u1, u1.conj()) - np.eye(M1))
Sigmainv[M1:, M1:] += (b ** 2 / (1 - b ** 2)) * (np.outer(u2, u2.conj()) - np.eye(M2))
Sigmainv[:M1, M1:] = -b * (1 / (1 - b ** 2)) * np.outer(u1, u2.conj())
Sigmainv[M1:,:M1] = Sigmainv[:M1, M1:].T.conj()

delta_inv = np.linalg.inv(Sigma) - Sigmainv

tr_ = np.trace(np.matmul(Sigmainv, C))
tr = 1/(1 - b**2) * ((M1 + M2) * (1 - b**2) + 2 * b**2 - 2 * np.real(rho) * b)

def negloglik(b):
    tr =  1/(1 - b**2) * ((M1 + M2) * (1 - b**2) + 2 * b**2 - 2 * np.real(rho) * b)
    ldet =  np.log(1 - b**2)
    return ldet + tr

b_test = np.linspace(0.01, 0.99, 1000)
nll = negloglik(b_test)
print(b_test[np.argmin(nll)])
import matplotlib.pyplot as plt
plt.plot(b_test, nll)
plt.show()
