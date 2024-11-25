# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:23:06 2024

@author: abhij
"""
import numpy as np
import matplotlib.pyplot as plt

def visualizer():
    data_file = 'Fs.npy'
    X = np.load(data_file).astype(np.float32)
    num_samples, num_timesteps, _, _ = X.shape
    sample = X[19, 90, 1:-1, :]
    for i in range(num_timesteps):
        plt.imshow(X[4, i, 1:-1, :], cmap='viridis')
        plt.show()
        

visualizer()