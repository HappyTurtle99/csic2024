import numpy as np
from helper_functions import *
from system_theta import system_theta
import sys
import os
import datetime

def run_for_theta(theta):
    system = system_theta(theta)
    return system.get_grad_at_cone(5)

def main():

    theta1 = float(sys.argv[1])
    theta2 = float(sys.argv[2])


    #this thetas is in degrees
    thetas = np.linspace(theta1, theta2, 100)
    alphas = np.zeros_like(thetas)


    #make sure to input in radians!
    renorm_vs = []
    for i, theta in enumerate(thetas):
        system = system_theta(np.radians(theta))
        renorm_vs.append(system.get_grad_at_cone(5))
        alphas[i] = system.alphasq

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