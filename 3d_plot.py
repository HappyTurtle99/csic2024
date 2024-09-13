import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits import mplot3d
from helper_functions import *

#returns matrix element of dirac hamiltonian for given k, 
# this is much cleaner to read in the function H
def d_entry(K, vf):
    return -1 * (K[0] + 1j * K[1]) #* 1.5 #* (1 - 9 * (0.14) ** 2)

def rotate(vec, theta):
    return np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), vec)

def H(kx, ky, Gs, deltaK, t, theta, vf, U=0):
    
    dk = np.array([kx, ky])
    G1, G2 = Gs
    G1_minus = rotate(G1, -theta/2)
    G1_plus = rotate(G1, theta/2)
    G2_minus = rotate(G2, -theta/2)
    G2_plus = rotate(G2, theta/2)

    ts = np.array([[t, t, t, t], [t, t * np.exp(1j * (2 * np.pi / 3)), t * np.exp(-1j * (2 * np.pi / 3)), t * np.exp(-1j * (2 * np.pi / 3))],
                   [t, t * np.exp(-1j * (2 * np.pi / 3)), t * np.exp(1j * (2 * np.pi / 3)), t * np.exp(1j * (2 * np.pi / 3))]])

    #ilegible but this is the UR of the hamiltonian 
    H_init = np.matrix([[0, d_entry(-deltaK / 2 + dk, vf), ts[1, 2], ts[1, 1], ts[2, 2], ts[2,1], ts[0, 0], ts[0, 0], 0, 0, 0, 0],
                       [0, 0, ts[1, 0], ts[1, 3], ts[2, 0], ts[2, 3], ts[0,0], ts[0,0], 0, 0, 0, 0],
                       [0, 0, 0, d_entry(deltaK / 2 + dk + G1_plus, vf), 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, d_entry(deltaK / 2 + dk + G1_plus + G2_plus, vf), 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, d_entry(deltaK / 2 + dk, vf), ts[1, 2].conjugate(), ts[1, 1].conjugate(), ts[2, 2].conjugate(), ts[2, 1].conjugate()],
                       [0, 0, 0, 0, 0, 0, 0, 0, ts[1, 0].conjugate(), ts[1, 3].conjugate(), ts[2, 0].conjugate(), ts[2, 3].conjugate()],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, d_entry(-deltaK / 2 + dk - G1_minus, vf), 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d_entry(-deltaK / 2 + dk - G1_minus - G2_minus, vf)],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
    
    V = np.zeros_like(H_init)
    V[0:2, 3:9] = U
    V[6:8, 6:12] = U

    H_init = H_init + V

    H_tot = H_init + H_init.H
    H_tot = H_tot
    eigvals = LA.eigvalsh(H_tot)
    return eigvals

#PARAMETROS
K = 4 * np.pi / 3
i = 31
t = 0.11
U = 0

theta = np.arccos((3 * i ** 2 + 3 * i + 0.5) / (3 * i ** 2 + 3 * i + 1))
deltaK = 2 * K * np.sin(theta / 2)
deltaK_vec = np.array([0, deltaK])
vf = 0.76 / deltaK

#DEFINE LOS VECTORES REALES Y RECIPROCOS
G1 = deltaK * np.array([np.sqrt(3)/2, 1/2])
G2 = deltaK * (np.array([-np.sqrt(3)/2, 1/2]))-G1
Gs = (G1, G2)

q0 = deltaK * np.array([0, -1])
q1 = deltaK * np.array([np.sqrt(3)/2, 1/2])
q2 = deltaK * np.array([-np.sqrt(3)/2, 1/2])
Qs = (q0, q1, q2)

H = populate_hamiltonian(0, 0, Qs, K, t, theta, vf, U)
N = (H.shape[0] / 4) ** 0.5
midband = H.shape[0] // 2 - 1

fig = plt.figure()
ax = plt.axes(projection='3d')

kx = np.linspace(-deltaK * 1.5, deltaK * 1.5, 100)
ky = np.linspace(-deltaK * 1.5, deltaK * 1.5, 100)

# kx = np.linspace(-1, 1 , 100)
# ky = np.linspace(-1, 1 / 2, 100)

kx, ky = np.meshgrid(kx, ky)

H_vals_hole = np.array([band_energy(kx[i, j], ky[i, j], Qs, K, t, theta, vf)[midband] for i in range(kx.shape[0]) for j in range(kx.shape[1])])
H_vals_elec = np.array([band_energy(kx[i, j], ky[i, j], Qs, K, t, theta, vf)[midband + 1] for i in range(kx.shape[0]) for j in range(kx.shape[1])])

# # Reshape H_vals to match the shape of kx and ky
H_vals_hole = H_vals_hole.reshape(kx.shape)
H_vals_elec = H_vals_elec.reshape(kx.shape)

# Plot the 3D surface
ax.plot_surface(kx, ky, H_vals_hole, cmap='viridis')
ax.plot_surface(kx, ky, H_vals_elec, cmap='viridis')

# Set labels and title
ax.set_xlabel('kx/deltaK')
ax.set_ylabel('ky/deltaK')
ax.set_zlabel('H')
ax.set_title('3D Plot of H')

print(H.shape)

# Show the plot
plt.show()
