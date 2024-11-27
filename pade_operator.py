
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class pade_neural_operator(nn.Module):
    
    def __init__(self, parameter_dim, latent_space_dim, pade_num_order, pade_denom_order, batch_size):
        
        super(pade_neural_operator, self).__init__()
        
        #fc layers
        self.fc_1 = 10
        self.fc_2 = 30
        self.fc_3 = 50
        self.fc_p_nc = pade_num_order                                #pade approximant numerator coefficients
        self.fc_p_np = pade_num_order                               #pade approximant numerator powers
        self.fc_p_dc = pade_denom_order                            #pade approximant denominator coefficients
        self.fc_p_dp = pade_denom_order
        
        #create the decoder
        self.fc1 = nn.Linear(in_features=parameter_dim, out_features=self.fc_1)
        self.fc2 = nn.Linear(in_features=self.fc_1, out_features=self.fc_2)
        self.fc3 = nn.Linear(in_features=self.fc_2, out_features=self.fc_3)
        self.fc4 = nn.Linear(in_features=self.fc_3, out_features=(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))

        
    def forward(self, x):
        
        x = x.view(batch_size, parameter_dim)
        
        #FC layers
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))
        x = f.tanh(self.fc3(x))
        x = f.tanh(self.fc4(x))
        
        return x

class FittingLoss(nn.Module):
    
    def __init__(self):
        super(FittingLoss, self).__init__()

    def forward(self, pade_vars, LS_vars, num_timesteps, pade_num_order, pade_denom_order, epsilon):
        
        time = torch.range(num_timesteps)
        prediction = torch.zeros(num_timesteps)
        
        num_coeffs = pade_vars[0:pade_num_order-1]
        num_powers = pade_vars[pade_num_order-1:2*pade_num_order-1]
        denom_coeffs = pade_vars[2*pade_num_order-1:(2*pade_num_order+pade_denom_order-1)]
        denom_powers = pade_vars[(2*pade_num_order+pade_denom_order-1):(2*pade_num_order+2*pade_denom_order-1)]
        
        numerator = num_coeffs*(time**num_powers)
        denominator = denom_coeffs*(time**denom_powers) + epsilon
        prediction = numerator/denominator
        

        loss = ((torch.mean((prediction - LS_vars)**2))**0.5)/torch.mean(LS_vars)
        
        return loss
    
#Latent space trajectory finder
if __name__ == "__main__":
    
    LS_data_file = "LS_dataset_train_data.npy"
    sample_data_file = "sample_data.npy"
    batch_size = 50
    
    
    print("Loading latent space dataset and sample dataset...")
    LS_data = np.load(LS_data_file).astype(np.float32)
    sample_data = np.load(sample_data_file).astype(np.float32)
    num_samples, num_timesteps, num_latent_space_vars = LS_data.shape
    _, parameter_dim = sample_data.shape
    
    for sample in range(num_samples - 3):
        plt.plot(range(num_timesteps), LS_data[sample, :, 2], color = 'k')
    plt.show()
    
    
    