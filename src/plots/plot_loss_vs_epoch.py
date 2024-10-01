import matplotlib.pyplot as plt
import numpy as np
import argparse

PATH = '/home/osvaldo/Documents/CCNY/Project_Saccades/'

parser = argparse.ArgumentParser(description='Plot Loss Function')
parser.add_argument('--exp', required=True, help='No experiment')
args = parser.parse_args()

evolution_folder = PATH +  'results/training/'
title = 'exp_' + args.exp

data = np.loadtxt(evolution_folder + title +'.dat')

plt.plot(data[0])
#plt.plot(data[1])
plt.show()