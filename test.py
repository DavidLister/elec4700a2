# test.py
#
# Starting point is :https://github.com/tjssmy/4700Code/blob/master/QMCode/SCE_SS.m
# This is just used to get more intuition behind the method
#

import numpy as np
import matplotlib.pyplot as plt

def pot_well(x, dx, x0, a, b):
    u = np.ones(x.shape)
    u = u * b
    u[x < x0 - dx] = a
    u[x > x0 + dx] = a
    return u

C_q_0 = 1.60217653e-19
C_hb = 1.054571596e-34
C_h = C_hb * 2 * np.pi
C_m_0 = 9.10938215e-31
C_kb = 1.3806504e-23
C_eps_0 = 8.854187817e-12
C_mu_0 = 1.2566370614e-6
C_c = 299792458

nx = 200
l = .40e-9
x = np.linspace(0, l, nx)
dx = x[1] - x[0]

pot = pot_well(x, l/4,l/2,75*C_q_0,0*C_q_0)

plt.plot(x, pot)
plt.show()

dx2 = dx**2
B = (C_hb**2) / (2 * C_m_0)

G = np.zeros((len(x), len(x)))
for i in range(len(x)):
    if i == 0:
        G[i][i] = B/dx2
    elif i == len(x) - 1:
        G[i][i] = B / dx2
    else:
        G[i][i] = 2 * B / dx2 + pot[i]
        G[i][i-1] = -B / dx2
        G[i][i+1] = -B / dx2

D, V = np.linalg.eig(G)
print(V.shape, D.shape)

idx = np.argsort(D)
V = V[:, idx]
D = D[idx]
print(V.shape, D.shape)
print(np.sum(D))


plt.plot(x * 1e9, pot/C_q_0/(5*70))
plt.plot(x * 1e9, V[:, 0:5])
superposition = np.sum(V[:, 0:5], axis=1)
plt.plot(x * 1e9, superposition)
plt.show()

