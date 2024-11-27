
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

#define the pade operator model
class Pade_Neural_Operator(nn.Module):
    
    def __init__(self, parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon):
        
        super(Pade_Neural_Operator, self).__init__()
        
        #fc layers
        self.fc_1 = 60
        self.fc_2 = 60
        self.fc_3 = 60
        
        self.fc_p_nc = pade_num_order                                #pade approximant numerator coefficients
        self.fc_p_np = pade_num_order                               #pade approximant numerator powers
        self.fc_p_dc = pade_denom_order                            #pade approximant denominator coefficients
        self.fc_p_dp = pade_denom_order                            #pade approximant denominator powers
        
        
        #create the decoder
        self.fc1 = nn.Linear(in_features=parameter_dim, out_features=self.fc_1)
        self.fc2 = nn.Linear(in_features=self.fc_1, out_features=self.fc_2)
        self.fc3 = nn.Linear(in_features=self.fc_2, out_features=self.fc_3)
        self.fc4 = nn.Linear(in_features=self.fc_3, out_features=(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))
        
        self.yscale = nn.Parameter(torch.ones(100))
        self.xscale = nn.Parameter(torch.ones(100))
        self.bias = nn.Parameter(torch.ones(100))
        
        
    def forward(self, x, time):
        
        x = x.view( batch_size, parameter_dim)
        
        #FC layers
        x1 = f.tanh(self.fc1(x))
        x2 = f.tanh(self.fc2(x1))
        x3 = f.tanh(self.fc3(x2))
        x4 = f.tanh(self.fc4(x3))
        
        #PNO layer
        num_coeffs = x4[:, 0:(self.fc_p_nc)]
        num_powers = pade_num_order*x4[:, self.fc_p_nc:(self.fc_p_nc + self.fc_p_np)]
        denom_coeffs = x4[:, (self.fc_p_nc + self.fc_p_np):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc)]
        denom_powers = pade_denom_order*x4[:, (self.fc_p_nc + self.fc_p_np + self.fc_p_dc):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dc)]
        
        time = time*self.xscale
        time_num = time.reshape(num_timesteps, 1)
        time_denom = time.reshape(num_timesteps, 1)
        time_num = time_num.repeat(1, pade_num_order)
        time_denom = time_denom.repeat(1, pade_denom_order)
        
        pade = torch.sum((num_coeffs*((time_num + epsilon)**num_powers)), dim = 1)/(torch.sum(((denom_coeffs*((time_denom + epsilon)**denom_powers))), dim =1)) 
        
        output = pade*self.yscale + self.bias
        
        return output

def train_model(pade_neural_operator, criterion, optimizer, sample_data, LS_data, run, learn_rate, LS_var_train_index, num_epochs=1, batchsize=20):
    
    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_timesteps = LS_data.shape
    _, parameter_dim = sample_data.shape
    
    
    # Convert to tensor and prepare DataLoader
    LS_tensor = torch.from_numpy(LS_data).float()  # Add channel dimension
    sample_tensor = torch.from_numpy(sample_data).float()
    time = torch.linspace(0, num_timesteps-1, num_timesteps)                      #time variable to create pade approximant
    
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        LS_tensor = LS_tensor.to(device)
        sample_tensor = sample_tensor.to(device)
        time = time.to(device)
    
    
    dataset = TensorDataset(sample_tensor, LS_tensor)
    
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.005)
    num_batches = len(dataloader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_num = 0
        
        for batch_inputs, batch_outputs in dataloader:
            
            # Get a batch of data
            sample = batch_inputs[0]  # Shape: (batchsize, 1, X_size, Y_size)
            LS_var = batch_outputs[0]
            
            # Forward pass
            prediction = pade_neural_operator(sample, time)
            
            
            if batch_num % 20 == 0 and epoch % 200 == 0: 
                pred = prediction.detach().cpu().numpy()
                LS_true = LS_var.detach().cpu().numpy()
                plt.plot(range(num_timesteps), pred, color = 'red')
                plt.plot(range(num_timesteps), LS_true, color = 'k')
                plt.show()
            
            loss = criterion(prediction, LS_var)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pade_neural_operator.parameters(), 0.2)
            
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            batch_num += 1
        
        # Scheduler step every 1000 epochs
        if epoch % 100 == 0 and epoch > 0:
            scheduler.step()
        
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    
        
        # Save model periodically
        if (epoch + 1) % 20000 == 0:
            print("Saving model...\n")
            torch.save(pade_neural_operator.state_dict(), "PNO_state_LSvar_"+str(LS_var_train_index)+"_run_"+str(run)+"_"+str(epoch)+".pth")


    
#Latent space trajectory finder
if __name__ == "__main__":
    
    LS_data_file = "LS_dataset_train_data.npy"              #dataset variable for the latent space (outputs)
    sample_data_file = "sample_data.npy"                    #dataset variable for the sample data (inputs)
    batch_size = 50                                         #batch size
    epsilon = 1e-4                                          #small coefficient for pade neural operator
    
    load_model = False
    restart_training = True
    use_CUDA = True
    run_to_load = 3
    epoch_to_load = 19999
    learn_rate = 1e-7
    batch_size = 1
    run = 4
    num_epochs = 100000
    
    #pade neural operator controls
    pade_num_order = 5
    pade_denom_order = 6
    LS_var_train_index = 2                                                              #index of latent space variable to train


    print("Loading latent space dataset and sample dataset...")
    LS_data = np.load(LS_data_file).astype(np.float32)
    sample_data = np.load(sample_data_file).astype(np.float32)
    num_samples, num_timesteps, num_latent_space_vars = LS_data.shape
    _, parameter_dim = sample_data.shape
    
    #show the latent space trajectories
    for sample in range(num_samples - 3):
        plt.plot(range(num_timesteps), LS_data[sample, :, LS_var_train_index], color = 'k')
    plt.show()
    
    #create pade neural operator 
    pade_neural_operator = Pade_Neural_Operator(parameter_dim, num_timesteps, pade_num_order, pade_denom_order, batch_size, epsilon)
    
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        print("CUDA available")
        pade_neural_operator = pade_neural_operator.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(pade_neural_operator.parameters())  , lr=learn_rate)

    LS_data_one_var = LS_data[:, :, LS_var_train_index]

    # Train the model
    if (load_model):
        print("Loading model...\n")
        pade_neural_operator.load_state_dict(torch.load("PNO_state_LSvar_"+str(LS_var_train_index)+"_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))

    elif (restart_training):
        print("Starting training with restart...\n")
        pade_neural_operator.load_state_dict(torch.load("PNO_state_LSvar_"+str(LS_var_train_index)+"_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        train_model(pade_neural_operator, criterion, optimizer, sample_data, LS_data_one_var, run, learn_rate, LS_var_train_index, num_epochs, batch_size)
    else:
        print("Starting training...\n")
        train_model(pade_neural_operator, criterion, optimizer, sample_data, LS_data_one_var, run, learn_rate, LS_var_train_index, num_epochs, batch_size)

    # Test with a new sample

    
        
    
    
    