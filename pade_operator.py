
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


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
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.05)

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
            
            #l2_reg = sum(p.pow(2.0).sum() for p in pade_neural_operator.parameters())
            
            
            loss = criterion(prediction, LS_var)/torch.var(LS_var)
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
        if epoch % 10000 == 0 and epoch > 0:
            scheduler.step()
        
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    
        
        # Save model periodically
        if (epoch + 1) % 10000 == 0:
            print("Saving model...\n")
            torch.save(pade_neural_operator.state_dict(), "PNO_state_LSvar_"+str(LS_var_train_index)+"_run_"+str(run)+"_"+str(epoch+1)+".pth")

def wasserstein_1d(x, y):
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))

    
#Latent space trajectory finder
if __name__ == "__main__":
    
    LS_data_file = "LS_dataset_train_data.npy"              #dataset variable for the latent space (outputs)
    sample_data_file = "sample_data.npy"                    #dataset variable for the sample data (inputs)
    batch_size = 50                                         #batch size
    epsilon = 1e-7                                          #small coefficient for pade neural operator
    
    load_model = False
    restart_training = False
    use_CUDA = True
    run_to_load = 11
    epoch_to_load = 10000
    LS_var_to_load = 2
    learn_rate = 1e-3
    batch_size = 20
    run = 12
    num_epochs = 200000
    
    #pade neural operator controls
    pade_num_order = 7
    pade_denom_order = 5
    LS_var_train_index = 0                                                            #index of latent space variable to train


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
        pade_neural_operator.load_state_dict(torch.load("PNO_state_LSvar_"+str(LS_var_to_load)+"_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))

    elif (restart_training):
        print("Starting training with restart...\n")
        pade_neural_operator.load_state_dict(torch.load("PNO_state_LSvar_"+str(LS_var_to_load)+"_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        train_model(pade_neural_operator, criterion, optimizer, sample_data, LS_data_one_var, run, learn_rate, LS_var_train_index, num_epochs, batch_size)
    else:
        print("Starting training...\n")
        train_model(pade_neural_operator, criterion, optimizer, sample_data, LS_data_one_var, run, learn_rate, LS_var_train_index, num_epochs, batch_size)

    # Test with a new sample

    
        
    
    
    