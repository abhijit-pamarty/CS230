
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r


    
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
        loss = torch.mean((predictions - targets)**2)
        return loss

# Generate synthetic data: Mapping from input matrix A
def generate_data(num_samples=100, input_size=5):
    # Random input-output matrices
    A = np.zeros((num_samples, input_size, input_size))
    A[:, 20:40, 20:40] = 1
    
    return torch.tensor(A, dtype=torch.float32)

# Model training
def train_model(encoder, decoder, criterion, optimizer, X, num_epochs=500):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = decoder(encoder(X))
        loss = criterion(output, X)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.2)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.2)
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Main function
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 100  # Number of features in input matrix
    input_channels = 1  # Number of features in output matrix
    latent_space_dim = 10  # Number of hidden units in the neural operator
    num_samples = 1 # Number of training samples

    # Create encoder and decoder
    encoder = Encoder(input_dim, input_channels, latent_space_dim)
    decoder = Decoder(input_dim, input_channels, latent_space_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters())  , lr=0.00001)

    # Generate data
    X = generate_data(num_samples, input_dim)

    # Train the model
    train_model(encoder, decoder, criterion, optimizer, X)

    # Test with a new sample
    test_sample = X[:, 1, 1]
    prediction = decoder(encoder(test_sample))
    print("Input matrix:", test_sample.numpy())
    print("Predicted output matrix:", prediction.detach().numpy())