

import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
import numpy as np


    
class NeuralOperator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(NeuralOperator, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.d1 = nn.Sequential(*layers)

    def forward(self, x):
        xfft = torch.real(fft.fft(x))
        xlinfft = self.d1(xfft)
        xifft = torch.real(fft.ifft(xlinfft))
        return torch.add(xifft, self.d1(x))
    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets)**2)
        return loss

# Generate synthetic data: Mapping from input matrix A to output matrix B
def generate_data(num_samples=100, input_size=5, output_size=5):
    # Random input-output matrices
    A = np.random.rand(num_samples, input_size)
    B = 2 * A + np.sin(A)  # Example transformation, replace with your own
    return torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)

# Model training
def train_model(model, criterion, optimizer, X, Y, num_epochs=500):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Main function
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 8  # Number of features in input matrix
    output_dim = 8  # Number of features in output matrix
    hidden_dim = 64  # Number of hidden units in the neural operator
    num_samples = 1000  # Number of training samples

    # Create model
    model = NeuralOperator(input_dim, hidden_dim, output_dim)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate data
    X, Y = generate_data(num_samples, input_dim, output_dim)

    # Train the model
    train_model(model, criterion, optimizer, X, Y)

    # Test with a new sample
    test_sample = torch.tensor([[0.5, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1]], dtype=torch.float32)
    true_matrix = 2 * test_sample.numpy() + np.sin(test_sample.numpy())
    prediction = model(test_sample)
    print("Input matrix:", test_sample.numpy())
    print("true output:", true_matrix)
    print("Predicted output matrix:", prediction.detach().numpy())
    print("Error: ", np.linalg.norm(prediction.detach().numpy() - true_matrix, ord = 'fro'))