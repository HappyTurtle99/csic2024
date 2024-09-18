import numpy as np
from helper_functions import *
from system_theta import system_theta
import sys
import os
import datetime


#really the input is alpha squared!! output is in radians
def get_theta(alpha):
    K = 4 * np.pi / 3
    vf = np.sqrt(3) * 2.7 / 2
    return 2 * np.arcsin(0.11/(2 * vf * K * np.sqrt(alpha)))

def main():

    #really alpha ** 2
    alpha1 = float(sys.argv[1])
    alpha2 = float(sys.argv[2])

    #this thetas is in degrees
    alphas = np.linspace(alpha1, alpha2, 100)

    #make sure to input in radians!
    renorm_vs = []
    for i, alpha in enumerate(alphas):
        system = system_theta(get_theta(alpha))
        renorm_vs.append(system.band_diff_at_cone())

    renorm_vs = np.array(renorm_vs)

    # Get the directory name
    directory_name = os.path.dirname(os.path.abspath(__file__))

    # Create a folder titled "alphas" with the current date and time and directory name
    folder_name = "alphas_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.path.basename(directory_name)
    os.mkdir(folder_name)

    # Save renorm_vs and alphas to the folder
    np.save(os.path.join(folder_name, 'renorm_vs.npy'), renorm_vs)
    np.save(os.path.join(folder_name, 'alphas.npy'), alphas)

    import matplotlib.pyplot as plt

    xs = np.linspace(0, 1, 100)
    # Plot renorm_vs against alpha
    plt.plot(alphas, renorm_vs)

    # Set labels for x and y axes
    plt.xlabel('alpha ** 2')
    plt.ylabel('vf*/vf (renormalized velocity)')

    # Save the plot
    plt.savefig(os.path.join(folder_name, 'renorm_vs_alpha.png'))

if __name__ == '__main__':
    main()