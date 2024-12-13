# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:11:17 2024

@author: abhij
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

####################################### AUTOENCODER ARCHITECTURE #######################################

    
class Encoder(nn.Module):
    
    def __init__(self, input_dim, input_channels, latent_space_dim, batch_size):
        
        super(Encoder, self).__init__()
        
        #encoder layers
        
        #channels in each convolution layer
        self.c_l0 = input_channels
        self.c_l1 = 100
        self.c_l2 = 200
        self.c_l3 = 500
        
        #image sizes after each convolution
        self.i_l1 = 31
        self.i_l2 = 15
        self.i_l3 = 7
        
        #filter size in convolution layer
        self.f_l1 = input_dim - self.i_l1 + 1
        self.f_l2 = self.i_l1 - self.i_l2 + 1
        self.f_l3 = self.i_l2 - self.i_l3 + 1
        
        #fc layers
        self.fc_1 = self.i_l3*self.i_l3*self.c_l3
        self.fc_2 = 200
        
        #create the encoder
        self.conv1 = nn.Conv2d(self.c_l0, self.c_l1, self.f_l1)
        self.conv2 = nn.Conv2d(self.c_l1, self.c_l2, self.f_l2)
        self.conv3 = nn.Conv2d(self.c_l2, self.c_l3, self.f_l3)
        self.fc1 = nn.Linear(in_features=self.fc_1, out_features=self.fc_2)
        self.fc2 = nn.Linear(in_features=self.fc_2, out_features=latent_space_dim)
        
        
    def forward(self, x):
        
        #conv layers
        x = f.tanh(self.conv1(x))
        x = f.tanh(self.conv2(x))
        x = f.tanh(self.conv3(x))
        
        #flatten
        x = x.view(batch_size, self.fc_1)
        
        #FC layers
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))

                
        return x

class Decoder(nn.Module):
    
    def __init__(self, input_dim, input_channels, latent_space_dim, batch_size):
        
        super(Decoder, self).__init__()
        
        #decoder layers
        
        #channels in each convolution layer
        self.c_l0 = input_channels
        self.c_l1 = 100
        self.c_l2 = 200
        self.c_l3 = 500
        
        #upsample rate
        self.scale1 = 4
        self.scale2 = 20
        
        #image sizes after each convolution
        self.i_l1 = 31
        self.i_l2 = 15
        self.i_l3 = 7
        
        #reshape dimension
        self.reshape_dim = (self.c_l3, 1, self.i_l3, self.i_l3)
        
        
        #fc layers
        self.fc_1 = self.i_l3*self.i_l3*self.c_l3
        self.fc_2 = 200
        
        #filter size in convolution layer
        self.f_l1 = self.i_l3*self.scale1 - self.i_l2 + 1
        self.f_l2 = self.i_l2*self.scale1 - self.i_l3 + 1
        self.f_l3 = self.i_l3*self.scale2 - input_dim + 1
        
        #create the decoder
        self.fc1 = nn.Linear(in_features=latent_space_dim, out_features=self.fc_2)
        self.fc2 = nn.Linear(in_features=self.fc_2, out_features=self.fc_1)
        self.upsample1 = nn.Upsample(scale_factor = self.scale1)
        self.upsample2 = nn.Upsample(scale_factor = self.scale2)
        self.conv1 = nn.Conv2d(self.c_l3, self.c_l2, self.f_l1)
        self.conv2 = nn.Conv2d(self.c_l2, self.c_l1, self.f_l2)
        self.conv3 = nn.Conv2d(self.c_l1, self.c_l0, self.f_l3)
        
    def forward(self, x):
        
        #unflatten
        x = x.view(batch_size, latent_space_dim)
        
        #FC layers
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))
        
        #reshape x
        x = torch.reshape(x, (batch_size, self.c_l3, self.i_l3, self.i_l3))
        
        #conv layers
        x = self.upsample1(x)
        x = f.tanh(self.conv1(x))
        
        x = self.upsample1(x)
        x = f.tanh(self.conv2(x))
        
        x = self.upsample2(x)
        x = f.tanh(self.conv3(x))
        x = torch.reshape(x, (batch_size, self.c_l0, input_dim, input_dim))
        
        return x

def generate_latent_space(encoder, X, latent_space_dim):
    
    #X organized as sample_index, time, X, Y
    
    
    num_samples, num_timesteps, _, _ = X.shape
    
    LS_dataset = np.zeros((num_samples, num_timesteps, latent_space_dim))
    print("Generating latent space data...")
    for sample_idx in range(num_samples):
        for timestep_idx in range(num_timesteps):
            print(f"Processing sample {sample_idx} at timestep {timestep_idx}...")
            test_sample = np.zeros((1, 1, 50, 50)).astype(np.float32)
            test_sample[0, 0, :, :] = X[sample_idx, timestep_idx, :, :]
            
            test_sample = torch.from_numpy(test_sample).float()
            
            if torch.cuda.is_available():
                print("CUDA available")
                device = torch.device("cuda:0")  # Specify the GPU device
                test_sample = test_sample.to(device)
                
            dataset = TensorDataset(test_sample)
            batchsize = 1  # Ensure batch size fits dataset
            dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
            LS = encoder(test_sample).cpu().detach().numpy()
            LS_dataset[sample_idx, timestep_idx, :] = LS
    
    return LS_dataset


####################################### PADE MODEL ARCHITECTURE #######################################  
#define the pade layer
#define the pade layer
class Pade_Layer(nn.Module):
    
    def __init__(self, parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon):
        
        super(Pade_Layer, self).__init__()
        
        #fc layers
        self.fc_1 = 20
        self.fc_2 = 20
        self.fc_3 = 20
        self.fc_4 = 20
        
        self.fc_p_nc = pade_num_order                                #pade approximant numerator coefficients
        self.fc_p_np = pade_num_order                               #pade approximant numerator powers
        self.fc_p_dc = pade_denom_order                            #pade approximant denominator coefficients
        self.fc_p_dp = pade_denom_order                            #pade approximant denominator powers
        
        
        #create the decoder
        self.fc1 = nn.Linear(in_features=parameter_dim, out_features=self.fc_1)
        self.fc2 = nn.Linear(in_features=self.fc_1, out_features=self.fc_2)
        self.fc3 = nn.Linear(in_features=self.fc_2, out_features=self.fc_3)
        self.fc4 = nn.Linear(in_features=self.fc_3, out_features=self.fc_4)
        self.fc5 = nn.Linear(in_features=self.fc_4, out_features=(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))
        

        
        
        self.fc6 = nn.Linear(in_features = parameter_dim, out_features = (self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))
        
        
    def forward(self, x, time):
        
        
        #FC layers
        x1 = f.leaky_relu(self.fc1(x))
        x2 = f.leaky_relu(self.fc2(x1))
        x3 = f.leaky_relu(self.fc3(x2))
        x4 = f.leaky_relu(self.fc4(x3))
        x5 = f.sigmoid(self.fc5(x4))
    
        
        #PNO layer
        num_coeffs = 2*(x5[ 0:(self.fc_p_nc)] - 0.5)
        num_powers = pade_num_order*x5[ self.fc_p_nc:(self.fc_p_nc + self.fc_p_np)]
        denom_coeffs = 2*(x5[(self.fc_p_nc + self.fc_p_np):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc)] - 0.5)
        denom_powers = pade_denom_order*x5[(self.fc_p_nc + self.fc_p_np + self.fc_p_dc):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dc)]
        
        time_num = time.reshape(num_timesteps, 1)
        time_denom = time.reshape(num_timesteps, 1)
        time_num = time_num.repeat(1, pade_num_order)
        time_denom = time_denom.repeat(1, pade_denom_order)
        
        pade = torch.sum((num_coeffs*((time_num + epsilon)**num_powers)), dim = 1)/(torch.sum(((denom_coeffs*((time_denom + epsilon)**denom_powers))), dim =1)) 
        
        short = torch.sum(f.tanh(self.fc6(x)))
        
        output = pade + short
        
        return output
    
class Pade_Neural_Operator(nn.Module):
    
    def __init__(self, parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon):
        
        super(Pade_Neural_Operator, self).__init__()
        
        #pade layers
        
        self.pade1 = Pade_Layer(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
        self.pade2 = Pade_Layer(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
        
        self.weights_layer = nn.Linear(in_features= parameter_dim, out_features= 1)
        
    def forward(self, x, time):
        
        
        #pade layers
        output1 = self.pade1(x, time)
        output2 = self.pade2(x, time)
        
        weights = self.weights_layer(x)
        
        return weights[0]*output1 + output2 





if __name__ == "__main__":
    
    ########################################### set up the pade neural operators ###########################################
    
    LS_data_file = "LS_dataset_train_data.npy"              #dataset variable for the latent space (outputs)
    sample_data_file = "sample_data.npy"                    #dataset variable for the sample data (inputs)
    batch_size = 50                                         #batch size
    epsilon = 1e-7                                          #small coefficient for pade neural operator
    
    load_model = False
    restart_training = True
    use_CUDA = True
    learn_rate = 5e-7
    batch_size = 1
    run = 12
    num_epochs = 200000
    
    #pade neural operator controls
    pade_num_order = 7
    pade_denom_order = 5          

    print("Loading latent space dataset and sample dataset...")
    LS_data = np.load(LS_data_file).astype(np.float32)
    sample_data = np.load(sample_data_file).astype(np.float32)
    num_samples, num_timesteps, num_latent_space_vars = LS_data.shape
    _, parameter_dim = sample_data.shape
    
    #create the instances of the model to load them
    pade_neural_operator_LS_0 = Pade_Neural_Operator(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
    pade_neural_operator_LS_1 = Pade_Neural_Operator(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
    pade_neural_operator_LS_2 = Pade_Neural_Operator(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
    pade_neural_operator_LS_3 = Pade_Neural_Operator(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
    
    #go to GPU
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        print("CUDA available")
        pade_neural_operator_LS_0 = pade_neural_operator_LS_0.to(device)
        pade_neural_operator_LS_1 = pade_neural_operator_LS_1.to(device)
        pade_neural_operator_LS_2 = pade_neural_operator_LS_2.to(device)
        pade_neural_operator_LS_3 = pade_neural_operator_LS_3.to(device)
        
    #Load the state of the pade operators
    print("loading state of pade neural operators")
    pade_neural_operator_LS_0.load_state_dict(torch.load("PNO_state_LSvar_0_final_2.pth"))
    pade_neural_operator_LS_1.load_state_dict(torch.load("PNO_state_LSvar_1_final_4.pth"))
    pade_neural_operator_LS_2.load_state_dict(torch.load("PNO_state_LSvar_2_final_2.pth"))
    pade_neural_operator_LS_3.load_state_dict(torch.load("PNO_state_LSvar_3_final_2.pth"))
    
    ########################################### set up the autoencoder ###########################################
    
    # Hyperparameters
    input_dim = 50  # Number of features in input matrix
    input_channels = 1  # Number of features in output matrix
    latent_space_dim = 4  # Number of units in latent space
    num_samples = 20 # Number of training samples
    run = 2
    total_epochs = 5000

    data_file = 'Fs.npy' 
    load_model = True
    restart_training = True
    use_CUDA = True
    run_to_load = 2
    epoch_to_load = 100
    learn_rate = 1e-5
    batch_size = 1
    
    # Create encoder and decoder
    encoder = Encoder(input_dim, input_channels, latent_space_dim, batch_size)
    decoder = Decoder(input_dim, input_channels, latent_space_dim, batch_size)
    
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        print("CUDA available")
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters())  , lr=learn_rate)

    # Generate data
    
    X = np.load(data_file).astype(np.float32)
    scaling_autoencoder = np.max(X)
    X_train = X/scaling_autoencoder
    
    # Load the model
    print("Loading model...\n")
    encoder.load_state_dict(torch.load("encoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
    decoder.load_state_dict(torch.load("decoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))

    # generate the latent space for the train data
    LS_dataset_train = generate_latent_space(encoder, X_train, latent_space_dim)

    
    ############################### LOAD TEST DATA AND PROCESS ############################### 
    
    test_sample_data_file = "sample_data_test.npy"
    test_data_file = "Fs_test.npy"
    
    X = np.load(test_data_file).astype(np.float32)
    X_test = X/scaling_autoencoder
    sample_data_test = np.load(test_sample_data_file).astype(np.float32)
    
    LS_dataset_test = generate_latent_space(encoder, X_test, latent_space_dim)
    
    ############################## Plot the errors ##############################
    
    ## Step 1 - comparison between autoencoder and true data
    
    X_autoencoder_train = np.zeros((20, 50, 50)).astype(np.float32)
    autoencoder_train_errors =  np.zeros((20, 50, 50)).astype(np.float32)
    autoencoder_test_errors =  np.zeros((3, 50, 50)).astype(np.float32)
    X_autoencoder_test =  np.zeros((3, 1, 50, 50)).astype(np.float32)
    
    #get the autoencoder results for the train dataset at the final timestep
    for train_data_index in range(20):
        
        test_sample = np.zeros((1, 1, 50, 50)).astype(np.float32)
        test_sample[0, 0, :, :] = X_train[train_data_index, 99, :, :]
        test_sample = torch.from_numpy(test_sample).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            test_sample = test_sample.to(device)
        
        dataset = TensorDataset(test_sample)
        batchsize = 1  # Ensure batch size fits dataset
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        
        for batch in dataloader:
            X_autoencoder_train[train_data_index, :, :] = decoder(encoder(batch[0])).cpu().detach().numpy()[0, 0, :, :]
            
        autoencoder_train_errors[train_data_index, :, :] = X_autoencoder_train[train_data_index, :, :] - X_train[train_data_index, 99, :, :]
        
    #get the autoencoder results for the test dataset at the final timestep
    for test_data_index in range(3):
        
        test_sample = np.zeros((1, 1, 50, 50)).astype(np.float32)
        test_sample[0, 0, :, :] = X_test[test_data_index, 99, :, :]
        test_sample = torch.from_numpy(test_sample).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            test_sample = test_sample.to(device)
        
        dataset = TensorDataset(test_sample)
        batchsize = 1  # Ensure batch size fits dataset
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        
        for batch in dataloader:
            X_autoencoder_test[test_data_index, :, :] = decoder(encoder(batch[0])).cpu().detach().numpy()[0, 0, :, :]
            
        autoencoder_test_errors[test_data_index, :, :] = X_autoencoder_test[test_data_index, :, :] - X_test[test_data_index, 99, :, :]
    
    #linear errors
    autoencoder_test_errors_lin = np.mean(autoencoder_test_errors.reshape(3, 2500), axis = 1)
    autoencoder_train_errors_lin = np.mean(autoencoder_train_errors.reshape(20, 2500), axis = 1)
    print("done")
    
    #step 2 comparison between pade approximant and true latent space
    
    pade_approximant_train = np.zeros((20, 100, 4)).astype(np.float32)
    pade_approximant_test = np.zeros((3, 100, 4)).astype(np.float32)
    pade_approximant_train_errors = np.zeros((20, 100, 4)).astype(np.float32)
    pade_approximant_test_errors = np.zeros((3, 100, 4)).astype(np.float32)
    
    # Convert to tensor and prepare DataLoader
    time = torch.linspace(0, num_timesteps-1, num_timesteps)
    
    for train_data_index in range(20):
        

        sample_tensor = torch.from_numpy(sample_data[train_data_index, :]).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            sample_tensor = sample_tensor.to(device)
            time = time.to(device)
        

        pade_approximant_train[train_data_index, :, 0] = pade_neural_operator_LS_0(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_train[train_data_index, :, 1] = pade_neural_operator_LS_1(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_train[train_data_index, :, 2] = pade_neural_operator_LS_2(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_train[train_data_index, :, 3] = pade_neural_operator_LS_3(sample_tensor, time).cpu().detach().numpy()
            
        pade_approximant_train_errors[train_data_index, :, :] = pade_approximant_train[train_data_index, :, :] - LS_dataset_train[train_data_index, :, :]
    
    for test_data_index in range(3):

        sample_tensor = torch.from_numpy(sample_data_test[test_data_index, :]).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            sample_tensor = sample_tensor.to(device)
            time = time.to(device)
        

        pade_approximant_test[test_data_index, :, 0] = pade_neural_operator_LS_0(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_test[test_data_index, :, 1] = pade_neural_operator_LS_1(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_test[test_data_index, :, 2] = pade_neural_operator_LS_2(sample_tensor, time).cpu().detach().numpy()
        pade_approximant_test[test_data_index, :, 3] = pade_neural_operator_LS_3(sample_tensor, time).cpu().detach().numpy()
            
        pade_approximant_test_errors[test_data_index, :, :] = pade_approximant_test[test_data_index, :, :] - LS_dataset_test[test_data_index, :, :]
    
    pade_test_errors_lin = np.mean(pade_approximant_test_errors, axis = 1)
    pade_train_errors_lin = np.mean(pade_approximant_train_errors, axis = 1)
    
    
    ###################################### pade approximant to final solution errors ######################################
    
    X_full_model_train = np.zeros((20, 50, 50)).astype(np.float32)
    full_model_train_errors =  np.zeros((20, 50, 50)).astype(np.float32)
    full_model_test_errors =  np.zeros((3, 50, 50)).astype(np.float32)
    X_full_model_test =  np.zeros((3, 1, 50, 50)).astype(np.float32)
    
    for train_data_index in range(20):
    
        sample_tensor = torch.from_numpy(pade_approximant_train[train_data_index, 99, :]).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            sample_tensor = sample_tensor.to(device)
        
        X_full_model_train[train_data_index, :, :] = decoder(sample_tensor).cpu().detach().numpy()[0, 0, :, :]
        full_model_train_errors[train_data_index, :, :] = X_full_model_train[train_data_index, :, :] - X_train[train_data_index, 99, :, :]

    for test_data_index in range(3):
    
        sample_tensor = torch.from_numpy(pade_approximant_test[test_data_index, 99, :]).float()
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify the GPU device
            sample_tensor = sample_tensor.to(device)
        
        X_full_model_test[test_data_index, :, :] = decoder(sample_tensor).cpu().detach().numpy()[0, 0, :, :]
        full_model_test_errors[test_data_index, :, :] = X_full_model_train[test_data_index, :, :] - X_train[test_data_index, 99, :, :]      
    
    #linear errors
    full_model_test_errors_lin = np.mean(full_model_test_errors.reshape(3, 2500), axis = 1)
    full_model_train_errors_lin = np.mean(full_model_train_errors.reshape(20, 2500), axis = 1)
    
    
    ################################## PLOT EVERYTHING ################################## 
    
    #### Autoencoder vs true data ####
    
    case_num = 0

    data1 = X_test[case_num, 99, :, :]
    data2 = X_autoencoder_test[case_num, 0, :, :]
    data3 = np.abs(autoencoder_test_errors[case_num, :, :])
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the images and add colorbars
    im1 = axs[0].imshow(data1, cmap='viridis')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('True concentration',fontsize=12)
    im1.set_clim(0, 1)
    
    im2 = axs[1].imshow(data2, cmap='viridis')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Reconstructed concentration',fontsize=12)
    im2.set_clim(0, 1)
    
    im3 = axs[2].imshow(np.log10(np.abs(data3*100)), cmap='inferno')
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title('Reconstruction error (log(e))',fontsize=12)
    im3.set_clim(-3, 2)


    plt.show()
    
    #### Autoencoder vs true data error graphs ####
    

    plt.scatter(range(20), np.log10(np.abs(autoencoder_train_errors_lin)*100), marker='x', color = 'k', label = "train data" )
    plt.scatter([20, 21, 22], np.log10(np.abs(autoencoder_test_errors_lin)*100), marker='+', color = 'red', label = "test data")
    legend = plt.legend()
    legend.get_frame().set_edgecolor('k')
    plt.ylim(-3,2)
    plt.title('Reconstruction error',fontsize=16)
    plt.xlabel("case index")
    plt.ylabel("log(e)")
    plt.show()
    
    
    #### Latent space variable graphs - train set#### 
    
    case_num = 7

    data1_1 = pade_approximant_train[case_num, :, 0] - np.min(pade_approximant_train[case_num, :, 0])
    data1_2 = LS_dataset_train[case_num, :, 0] - np.min(pade_approximant_train[case_num, :, 0])
    data2_1 = pade_approximant_train[case_num, :, 1] - np.min(pade_approximant_train[case_num, :, 1])
    data2_2 = LS_dataset_train[case_num, :, 1] - np.min(pade_approximant_train[case_num, :, 1])
    data3_1 = pade_approximant_train[case_num, :, 2] - np.min(pade_approximant_train[case_num, :, 2])
    data3_2 = LS_dataset_train[case_num, :, 2] - np.min(pade_approximant_train[case_num, :, 2])
    data4_1 = pade_approximant_train[case_num, :, 3] - np.min(pade_approximant_train[case_num, :, 3])
    data4_2 = LS_dataset_train[case_num, :, 3] - np.min(pade_approximant_train[case_num, :, 3])
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    
    axs[0, 0].plot(range(num_timesteps), data1_1, color = 'k')
    axs[0, 0].plot(range(num_timesteps), data1_2, color = 'red')
    
    axs[0, 1].plot(range(num_timesteps), data2_1, color = 'k')
    axs[0, 1].plot(range(num_timesteps), data2_2, color = 'red')
    
    axs[1, 0].plot(range(num_timesteps), data3_1, color = 'k')
    axs[1, 0].plot(range(num_timesteps), data3_2, color = 'red')
    
    axs[1, 1].plot(range(num_timesteps), data4_1, color = 'k')
    axs[1, 1].plot(range(num_timesteps), data4_2, color = 'red')
    fig.suptitle('Latent space variable predictions - train set',fontsize=16)
    
    axs[0, 0].axis(ymin = 0, ymax = 0.7)
    axs[0, 1].axis(ymin = 0, ymax = 0.7)
    axs[1, 0].axis(ymin = 0, ymax = 0.7)
    axs[1, 1].axis(ymin = 0, ymax = 0.7)
    
    axs[0, 0].axis(xmin = 0, xmax = 99)
    axs[0, 1].axis(xmin = 0, xmax = 99)
    axs[1, 0].axis(xmin = 0, xmax = 99)
    axs[1, 1].axis(xmin = 0, xmax = 99)
    plt.show()
    
    #### Latent space variable graphs - test set#### 
    
    case_num = 0

    data1_1 = pade_approximant_test[case_num, :, 0] - np.min(pade_approximant_test[case_num, :, 0])
    data1_2 = LS_dataset_test[case_num, :, 0] - np.min(pade_approximant_test[case_num, :, 0])
    data2_1 = pade_approximant_test[case_num, :, 1] - np.min(pade_approximant_test[case_num, :, 1])
    data2_2 = LS_dataset_test[case_num, :, 1] - np.min(pade_approximant_test[case_num, :, 1])
    data3_1 = pade_approximant_test[case_num, :, 2] - np.min(pade_approximant_test[case_num, :, 2])
    data3_2 = LS_dataset_test[case_num, :, 2] - np.min(pade_approximant_test[case_num, :, 2])
    data4_1 = pade_approximant_test[case_num, :, 3] - np.min(pade_approximant_test[case_num, :, 3])
    data4_2 = LS_dataset_test[case_num, :, 3] - np.min(pade_approximant_test[case_num, :, 3])
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    
    axs[0, 0].plot(range(num_timesteps), data1_1, color = 'k')
    axs[0, 0].plot(range(num_timesteps), data1_2, color = 'red')
    
    axs[0, 1].plot(range(num_timesteps), data2_1, color = 'k')
    axs[0, 1].plot(range(num_timesteps), data2_2, color = 'red')
    
    axs[1, 0].plot(range(num_timesteps), data3_1, color = 'k')
    axs[1, 0].plot(range(num_timesteps), data3_2, color = 'red')
    
    axs[1, 1].plot(range(num_timesteps), data4_1, color = 'k')
    axs[1, 1].plot(range(num_timesteps), data4_2, color = 'red')
    fig.suptitle('Latent space variable predictions - test set',fontsize=16)

    axs[0, 0].axis(ymin = 0, ymax = 0.7)
    axs[0, 1].axis(ymin = 0, ymax = 0.7)
    axs[1, 0].axis(ymin = 0, ymax = 0.7)
    axs[1, 1].axis(ymin = 0, ymax = 0.7)
    
    axs[0, 0].axis(xmin = 0, xmax = 99)
    axs[0, 1].axis(xmin = 0, xmax = 99)
    axs[1, 0].axis(xmin = 0, xmax = 99)
    axs[1, 1].axis(xmin = 0, xmax = 99)
    plt.show()
    
    #### Latent space variable errors #### 
    
    plt.scatter(range(20), np.log10(np.abs(pade_train_errors_lin[:, 0])*100), marker='x', color = 'k', label = "train data - LS0" )
    plt.scatter([20, 21, 22], np.log10(np.abs(pade_test_errors_lin[:, 0])*100), marker='+', color = 'red', label = "test data - LS0")
    plt.scatter(range(20), np.log10(np.abs(pade_train_errors_lin[:, 1])*100), marker='*', color = 'k', label = "train data - LS1" )
    plt.scatter([20, 21, 22], np.log10(np.abs(pade_test_errors_lin[:, 1])*100), marker='o', color = 'red', label = "test data - LS1")
    plt.scatter(range(20), np.log10(np.abs(pade_train_errors_lin[:, 2])*100), marker='s', color = 'k', label = "train data - LS2" )
    plt.scatter([20, 21, 22], np.log10(np.abs(pade_test_errors_lin[:, 2])*100), marker='v', color = 'red', label = "test data - LS2")
    plt.scatter(range(20), np.log10(np.abs(pade_train_errors_lin[:, 3])*100), marker='<', color = 'k', label = "train data - LS3" )
    plt.scatter([20, 21, 22], np.log10(np.abs(pade_test_errors_lin[:, 3])*100), marker='d', color = 'red', label = "test data - LS3")
    #legend = plt.legend()
    plt.ylim(-3,2)
    plt.xlabel("case index")
    plt.ylabel("log(e)")
    #legend.get_frame().set_edgecolor('k')
    plt.title('Latent space variable prediction error',fontsize=16)
    
    #### Full model vs true data ####
    
    case_num = 0

    data1 = X_test[case_num, 99, :, :]
    data2 = X_full_model_test[case_num, 0, :, :]
    data3 = np.abs(full_model_test_errors[case_num, :, :])
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the images and add colorbars
    im1 = axs[0].imshow(data1, cmap='viridis')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('True concentration',fontsize=12)
    im1.set_clim(0, 1)
    
    im2 = axs[1].imshow(data2, cmap='viridis')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Model predicted concentration',fontsize=12)
    im2.set_clim(0, 1)
    
    im3 = axs[2].imshow(np.log10(np.abs(data3*100)), cmap='inferno')
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title('Prediction error (log(e))',fontsize=12)
    im3.set_clim(-3, 2)

    plt.show()
    
    #### full model vs true data error graphs ####
    

    plt.scatter(range(20), np.log10(np.abs(full_model_train_errors_lin)*100), marker='D', color = 'k', label = "train data" )
    plt.scatter([20, 21, 22], np.log10(np.abs(full_model_test_errors_lin)*100), marker='p', color = 'red', label = "test data")
    legend = plt.legend()
    legend.get_frame().set_edgecolor('k')
    plt.title('Prediction error',fontsize=16)
    plt.ylim(-3,2)
    plt.xlabel("case index")
    plt.ylabel("log(e)")

    plt.show()
    
    
    
    