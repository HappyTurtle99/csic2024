import numpy as np
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits import mplot3d

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

    H = band_energy(0, 0, Gs, deltaK_vec, t, theta, vf, U=0)
