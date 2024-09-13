from numba import jit
import numpy as np
import numpy.linalg as LA
import sys

@jit(nopython=True)
def d_entry(K, vf):
    return -1 * (K[0] + 1j * K[1]) * vf

# @jit(nopython=True) 
#SHOULD REALLY MAKE NUMBA COMPATIBLE, BUT THE POPULATION IS NOT THAT BAD
def populate_hamiltonian(kx, ky, Qs, deltaK_vec, t, theta, vf, N=None, U=0):
    
    dk = np.array([kx, ky])

    q0, q1, q2 = Qs
    a = q1 - q2
    b = q1 - q0

    if not N:
        #Calculate number of unit cells:
        N = 2 * int(np.radians(3.9) / theta / 1.5) + 1

        if N == 1:
            N = 3
    
    #Initialize Hamiltonian
    H_init = np.zeros((4 * N ** 2, 4 * N ** 2), dtype=np.complex128)

    ts = np.array([[t, t, t, t], [t, t * np.exp(1j * (2 * np.pi / 3)), t * np.exp(-1j * (2 * np.pi / 3)), t * np.exp(-1j * (2 * np.pi / 3))],
                   [t, t * np.exp(-1j * (2 * np.pi / 3)), t * np.exp(1j * (2 * np.pi / 3)), t * np.exp(1j * (2 * np.pi / 3))]])

    #Populate Everything but the first and last columns and rows
    # of the Hamiltonian for sublattice 1.
    #iterate over momentum lattice, then find corresponding row in Hamiltonian
    #for each state and coupled states
    for i in range((N - 1)):
        for j in range((N - 1)):
            #index of k_row within momentum grid
            k_index = (i + 1) * N + j + 1
            #corresponding row in Hamiltonian
            n = k_index * 4

            #index of the k that this row is coupled to
            k1 = k_index
            k2 = k_index - N
            k3 = k_index - N - 1

            #corresponding columns in Hamiltonian
            ms = [2 + ki * 4 for ki in [k1, k2, k3]]

            #calculate center coordinates (here we add 1 to i and j 
            # because we start counting from the 2nd row and column)
            center = (j + 1 - (N - 1) / 2) * a - (i + 1 - (N - 1) / 2) * b

            H_init[n, n + 1] = d_entry(dk + center, vf)
            
            for k, m in enumerate(ms):
                if m >= n:
                    T = np.array([[ts[k, 2], ts[k, 1]], [ts[k, 0], ts[k, 3]]])
                    H_init[n:n+2, m:m+2] = T
                elif m < n:
                    T = np.array([[ts[k, 2].conj(), ts[k, 1].conj()], [ts[k, 0].conj(), ts[k, 3].conj()]])
                    H_init[n:n+2, m:m+2] = T

    #populate the elements corresponding to the edge of the momentum grid
    #that have not been included. these are not connected to every state
    #sublattice 1 these edges are bottom and right
    for i in range(N):
        #top
        k_index = i
        n = k_index * 4
        
        k1 = k_index

        m = k1 * 4 + 2

        #diagonal element
        center = (i - (N - 1) / 2) * a + ((N - 1) / 2) * b
        H_init[n, n + 1] = d_entry(dk + center, vf)
        #off diagonal elements
        T = np.array([[ts[0, 2], ts[0, 1]], [ts[0, 0], ts[0, 3]]])
        H_init[n:n+2, m:m+2] = T

        #left
        if i != 0:
            k_index = N * (i - 1) + N
            n = k_index * 4
            
            k1 = k_index
            k2 = k_index - N
            ms = [2 + ki * 4 for ki in [k1, k2]]

            #diagonal element
            center = -((N - 1) / 2) * a - (i - (N - 1) / 2) * b
            H_init[n, n + 1] = d_entry(dk + center, vf)

            #off diagonal elements
            for k, m in enumerate(ms):
                if m > n:
                    T = np.array([[ts[k, 2], ts[k, 1]], [ts[k, 0], ts[k, 3]]])
                    H_init[n:n+2, m:m+2] = T
                elif m < n:
                    T = np.array([[ts[k, 2].conj(), ts[k, 1].conj()], [ts[k, 0].conj(), ts[k, 3].conj()]])
                    H_init[n:n+2, m:m+2] = T

    #off diagonal elements have all been included in H + H.conj().T 
    #since the hopping is symmetric
    #populate the diagonal elements of sublattice 2

    for i in range(N ** 2):
        k_index = i
        n = k_index * 4 + 2
        center = (i % N - (N - 1) / 2) * a - (i // N - (N - 1) / 2) * b
        H_init[n, n + 1] = d_entry(q0 + dk + center, vf)
    
    H = H_init + H_init.conj().T
    return H

def band_energy(kx, ky, Qs, deltaK, t, theta, vf, N=None, U=0):
    H = populate_hamiltonian(kx, ky, Qs, deltaK, t, theta, vf, N=N, U=U)
    return LA.eigvalsh(H)

@jit(nopython=True) 
def calculate_bands():
    return

def generate_vectors(A, B, C, N):
    A_hat = A / np.linalg.norm(A)
    step_size = C / (N - 1)
    
    vectors = np.array([B - (C / 2) * A_hat + i * step_size * A_hat for i in range(N)])
    
    return vectors

def generate_path(N, pt1, pt2):
    diff = pt2 - pt1
    step_size = np.linalg.norm(diff) / (N - 1)
    vectors = np.array([pt1 + i * step_size * diff / np.linalg.norm(diff) for i in range(N)])
    return vectors

#generate the path in momentum space used in macdonald's paper
def generate_macpath(n_pts, deltaK, Qs, starting_cell = (0, 0)):

    ks = np.zeros((n_pts, 2))

    #in units of deltaK of ABCDA
    tot_len = 1 + 1 + np.sqrt(3) + 1

    n, m = starting_cell
    q0, q1, q2 = Qs
    a = q1 - q2
    b = q1 - q0
    starting_vec = n * a + m * b

    pt1 = np.array([0, -deltaK]) + starting_vec
    pt2 = np.array([0, deltaK]) + starting_vec
    pt3 = deltaK * np.array([np.sqrt(3)/2, -1/2]) + starting_vec
    N = int(n_pts * 2 / tot_len)
    ks_ABC = generate_path(N, pt1, pt2)
    N = int(n_pts * np.sqrt(3) / tot_len)
    ks_CD = generate_path(N, pt2, pt3)
    N = int(n_pts / tot_len)
    ks_DA = generate_path(N, pt3, pt1)

    ks[:len(ks_ABC)] = ks_ABC
    ks[len(ks_ABC):len(ks_ABC) + len(ks_CD)] = ks_CD
    ks[len(ks_ABC) + len(ks_CD):len(ks_ABC)+len(ks_CD)+len(ks_DA)] = ks_DA

    return ks

def grad(E, x, deltaK):
    grad = np.zeros_like(E)
    unit = deltaK * (3 + np.sqrt(3)) / len(x)

    for i in range(1, len(E) - 1):
        grad[i] = (E[i + 1] - E[i - 1]) / unit

    grad[0] = (E[1] - E[0]) / unit
    grad[-1] = (E[-1] - E[-2]) / unit

    return grad

def grad_at_cone(E, x, deltaK):
    cone_index = int(len(x) * 1/(3+np.sqrt(3)))
    g = grad(E, x, deltaK)
    return g[cone_index]

#THIS IS NOT A SCRIPT, IT IS A MODULE. THIS IS FOR DEBUGGING:
if __name__ == '__main__':

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
    a1 = np.array([1, np.sqrt(3)]) / 2
    a2 = np.array([-1, np.sqrt(3)]) / 2
    # G1 = (4 * np.pi) * ((3 * i + 1) * a1 + a2) / (3 * (3 * i ** 2 + 3 * i + 1))
    # G2 = (4 * np.pi) * (-(3 * i + 2) * a1 + (3 * i + 1) * a2) / (3 * (3 * i ** 2 + 3 * i + 1))
    # Gs = (G1, G2)

    G1 = deltaK * np.array([np.sqrt(3)/2, 1/2])
    G2 = deltaK * (np.array([-np.sqrt(3)/2, 1/2]))-G1
    Gs = (G1, G2)

    q0 = deltaK * np.array([0, -1])
    q1 = deltaK * np.array([np.sqrt(3)/2, 1/2])
    q2 = deltaK * np.array([-np.sqrt(3)/2, 1/2])
    Qs = (q0, q1, q2)

    np.set_printoptions(threshold = sys.maxsize)

    H = populate_hamiltonian(0, 0, Qs, deltaK_vec, t, theta, vf, U=0)
    ks = generate_macpath(100, deltaK)
