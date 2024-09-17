import numpy as np
from helper_functions import *

class system_theta():
    def __init__(self, theta, N=None, U=None):
        
        #get deltaK exactly:
        self.K = 4 * np.pi / 3
        i_santos = 8
        theta_temp = np.arccos((3 * i_santos ** 2 + 3 * i_santos + 0.5) / (3 * i_santos ** 2 + 3 * i_santos + 1))
        deltaK = 2 * self.K * np.sin(theta_temp / 2)
        self.vf = 0.76 / deltaK

        #specify other params
        self.t = 0.11
        self.U = U
        
        self.theta = theta
        self.deltaK = 2 * self.K * np.sin(self.theta / 2)
        deltaK_vec = np.array([0, self.deltaK])

        self.alphasq = (self.t / (self.vf * self.deltaK)) ** 2

        q0 = self.deltaK * np.array([0, -1])
        q1 = self.deltaK * np.array([np.sqrt(3)/2, 1/2])
        q2 = self.deltaK * np.array([-np.sqrt(3)/2, 1/2])
        self.Qs = (q0, q1, q2)


        #NB WE ARE ASSUME THETA IS IN Radians
        if N is not None: 
            self.N = N
        else:
            self.N = 2 * int(np.radians(3.9) / self.theta / 1.5) + 1
            if self.N == 1:
                self.N = 3

        self.midband = 4 * self.N ** 2 // 2 - 1

    #macpath assumed
    def get_bands(self, number_of_bands, window):
        ks = generate_macpath(100, self.deltaK, self.Qs, starting_cell=(0, 0))
        cone_index = int(len(ks) * 1/(3+np.sqrt(3)))
        kxs = ks[cone_index-window:cone_index+window, 0]
        kys = ks[cone_index-window:cone_index+window, 1]
        
        xs = np.linspace(-1, 1, len(kxs))
        Es = np.array([band_energy(kxs[i], kys[i], self.Qs, self.K, self.t, self.theta, self.vf, U=self.U, N=self.N) for i in range(len(kxs))])

        slice = len(xs) / len(ks)

        return Es[:,self.midband - number_of_bands // 2:self.midband + number_of_bands // 2], xs, slice
    
    def get_grad_at_cone(self, window):
        Es, xs, slice = self.get_bands(2, window)
        return grad_at_cone(Es[:,0], xs, slice, self.deltaK) / self.vf

