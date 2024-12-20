
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

    
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
  
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        
        loss = ((torch.mean((predictions - targets)**2))**0.5)/torch.mean(targets)
        return loss

def train_model(encoder, decoder, criterion, optimizer, X, run, learn_rate, num_epochs=1, batchsize=50):
    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_timesteps, X_size, Y_size = X.shape
    X_reshaped = X.reshape(num_samples * num_timesteps, X_size, Y_size)
    
    # Convert to tensor and prepare DataLoader
    X_tensor = torch.from_numpy(X_reshaped).float().unsqueeze(1)  # Add channel dimension
    
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        X_tensor = X_tensor.to(device)
    
    
    dataset = TensorDataset(X_tensor)
    
    batchsize = min(batchsize, len(dataset))  # Ensure batch size fits dataset
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.1)
    num_batches = len(dataloader)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_num = 0


        
        for batch in dataloader:
            # Get a batch of data
            sample = batch[0]  # Shape: (batchsize, 1, X_size, Y_size)
            
            # Forward pass
            encoded = encoder(sample)
            output = decoder(encoded)
            loss = criterion(output, sample)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.2)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.2)
            
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            print(f"Processing batch {batch_num+1}/{num_batches}")
        
        # Scheduler step every 1000 epochs
        if epoch % 10 == 0 and epoch > 0:
            scheduler.step()
        
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        
        # Visualize error every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                error = sample[0, 0, :, :].detach().cpu().numpy() - output[0, 0, :, :].detach().cpu().numpy()
                #plt.imshow(error, cmap='hot', interpolation='nearest')
                #plt.title(f"Error Visualization (Epoch {epoch+1})")
                #plt.colorbar()
                #plt.show()
        
        # Save model periodically
        if epoch % 25 == 0:
            print("Saving model...\n")
            torch.save(encoder.state_dict(), f"encoder_state_run_{run}_{epoch}.pth")
            torch.save(decoder.state_dict(), f"decoder_state_run_{run}_{epoch}.pth")

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
    
# Main function
if __name__ == "__main__":
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
    X = X/np.max(X)
    X_train = X[:, :, :, :]
    X_test = X[17:19, :, :, :]

    # Train the model
    if (load_model):
        print("Loading model...\n")
        encoder.load_state_dict(torch.load("encoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        decoder.load_state_dict(torch.load("decoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
    elif (restart_training):
        print("Starting training with restart...\n")
        encoder.load_state_dict(torch.load("encoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        decoder.load_state_dict(torch.load("decoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        train_model(encoder, decoder, criterion, optimizer, X_train, run, learn_rate, total_epochs)
    else:
        print("Starting training...\n")
        train_model(encoder, decoder, criterion, optimizer, X_train, run, learn_rate, total_epochs)

    # Test with a new sample

    LS_dataset = generate_latent_space(encoder, X, latent_space_dim)
    np.save('LS_dataset_train_data.npy', LS_dataset)
    
    """
    test_sample = np.zeros((1, 1, 50, 50)).astype(np.float32)
    test_sample[0, 0, :, :] = X[19, 50, :, :]

    
    test_sample = torch.from_numpy(test_sample).float()
    
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        test_sample = test_sample.to(device)
    
    dataset = TensorDataset(test_sample)
    batchsize = 1  # Ensure batch size fits dataset
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    prediction = []
    for batch in dataloader:
        prediction = decoder(encoder(batch[0]))
    plt.imshow(test_sample.cpu().numpy()[0, 0, :, :])
    plt.title("true data")
    plt.show()
    plt.imshow(prediction.detach().cpu().numpy()[0, 0, :, :])
    plt.title("predicted data")
    plt.show()
    """
    
    