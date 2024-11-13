
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt

    
class Encoder(nn.Module):
    
    def __init__(self, input_dim, input_channels, latent_space_dim):
        
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
        x = x.view(-1)
        
        #FC layers
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))

                
        return x

class Decoder(nn.Module):
    
    def __init__(self, input_dim, input_channels, latent_space_dim):
        
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
        
        
        
        #FC layers
        x = f.tanh(self.fc1(x))
        x = f.tanh(self.fc2(x))
        
        #reshape x
        x = torch.reshape(x, (1, self.c_l3, self.i_l3, self.i_l3))
        
        #conv layers
        x = self.upsample1(x)
        x = f.tanh(self.conv1(x))
        
        x = self.upsample1(x)
        x = f.tanh(self.conv2(x))
        
        x = self.upsample2(x)
        x = f.tanh(self.conv3(x))
        x = torch.reshape(x, (input_dim, input_dim))
        
        return x
  
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = ((torch.mean((predictions - targets)**2))**0.5)/torch.mean(targets)
        return loss

# Generate synthetic data: Mapping from input matrix A
def generate_data(num_samples=100, input_size=5):
    # Random input-output matrices
    A = np.zeros((num_samples, input_size, input_size))
    A[:, 20:40, 20:40] = 1
    
    return torch.tensor(A, dtype=torch.float32)

# Model training
def train_model(encoder, decoder, criterion, optimizer, X, run, num_epochs=1):
    
    num_samples, _, _ = X.shape
    for epoch in range(num_epochs):
        
        
        #with SGD
        sample_num = r.randint(0, num_samples - 1)
        sample = np.zeros((1, 1, 100, 100)).astype(np.float32)
        sample[0, 0, :, :] = X[sample_num, :, :]
        sample = torch.from_numpy(sample)
        optimizer.zero_grad()
        output = decoder(encoder(sample))
        loss = criterion(sample, output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.2)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.2)
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            
        if epoch % 500 == 0:
            print("Saving model...\n")
            torch.save(encoder.state_dict(), "encoder_state_run_"+str(run)+"_"+str(epoch)+".pth")
            torch.save(decoder.state_dict(), "decoder_state_run_"+str(run)+"_"+str(epoch)+".pth")

# Main function
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 100  # Number of features in input matrix
    input_channels = 1  # Number of features in output matrix
    latent_space_dim = 4  # Number of units in latent space
    num_samples = 20 # Number of training samples
    run = 1
    total_epochs = 2000

    data_file = 'Taus.npy'
    load_model = True
    run_to_load = 1
    epoch_to_load = 1000
    
    
    # Create encoder and decoder
    encoder = Encoder(input_dim, input_channels, latent_space_dim)
    decoder = Decoder(input_dim, input_channels, latent_space_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters())  , lr=0.00001)

    # Generate data
    X = np.load(data_file).astype(np.float32)
    X = X/np.max(X)

    # Train the model
    if (load_model):
        print("Loading model...\n")
        encoder.load_state_dict(torch.load("encoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        decoder.load_state_dict(torch.load("decoder_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
    else:
        print("Starting training...\n")
        train_model(encoder, decoder, criterion, optimizer, X, run, total_epochs)

    # Test with a new sample

    test_sample = np.zeros((1, 1, 100, 100)).astype(np.float32)
    test_sample[0, 0, :, :] = X[1, :, :]
    test_sample = torch.from_numpy(test_sample)
    prediction = decoder(encoder(test_sample))
    print("Input matrix:", test_sample.numpy())
    plt.imshow(test_sample.numpy()[0, 0, :, :])
    plt.title("true data")
    plt.show()
    plt.imshow(prediction.detach().numpy())
    plt.title("predicted data")
    plt.show()
    print("Predicted output matrix:", prediction.detach().numpy())