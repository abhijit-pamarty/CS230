# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:13:30 2024

@author: wygli
"""

# This code uses the Cottrell Equation for unidirection diffusion to estimate Lithium Ion concentrations.
# Data is taken from the DIB dataset from the University of Warwick.

import torch
import torchvision
import numpy as np
import datetime, os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
#import deepxde as dde
import numpy as np
#from deepxde.backend import tf

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
    
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets).squeeze()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# This function takes in a folder path with the DIB csv files and returns a list of dataframes with data
# from the CSV files
def getCSV(folder_path):
    files = []
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for file in csv_files:
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(folder_path, file)
        dataframe = pd.read_csv(file_path, skiprows=range(15))
        files.append(dataframe)
    return files

#This function defines the Cottrell Equation which is Fick's second law of diffusion solved
# for specific boundary conditions for a planar electrode
def Cottrell_eq(t,n,D,F,A,i):
    return i/(n*F*A*(D**(1/2))/((np.pi**(1/2))*t**(1/2)))

def Nernst_eq(E,E0, n, F, R, T):
    #return U+((R*T/(n*F))*np.log((1-x)/x))+V_ni
    #return 1/(1+np.exp(-n*F*(E-E0)/(R*T))) #return x, concentration of reduced species
    return np.exp(E0-E)*(n*F/(R*T)) #return Q, [red]/[ox], = [li+][C6]/[LiC6] at anode

F = 96485 #Faraday's constant, C/mol
n = 1 #number of electrons transferred per ion
A = 1 #surface area of electrode
D = 2.5*10**(-6) #diffusion coefficient of LiPF6 in EC:DMC, cm^2/s
R = 8.314 #gas constant
T = 25+273.15 #temperature
E0 = 3.7
    

folder_path=r"C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\DIB_Data\.csvfiles\Capacity_Check\80per_Cells_Capacity_Check_08122021_080cycle"
files = getCSV(folder_path)

# Calculating concentration at surface of electrode using Cottrell  Equation
#surface_conc = Cottrell_eq(files[0].iloc[40:19000, 3].astype(float), n, D, F, A, abs(files[0].iloc[40:19000, 8].astype(float)))
#fig1 = plt.figure(figsize=(10, 5))
#plt.ylabel('Surface Concentrations')
#plt.plot(surface_conc)
#plt.grid()

# Calculating total concentration in electrode using Nernst equation
fig2 = plt.figure(figsize=(10, 5))
Q=[]
li_conc=[]
for i in range(len(files)):
    Q.append(Nernst_eq(files[i].iloc[40:19000,7].astype(float), E0, n, F, R,T))
    lic6_conc = Q[i].iloc[0]
    c6_conc = 1
    li_conc.append(Q[i]*c6_conc/lic6_conc)
    plt.plot(li_conc[i])


plt.ylabel('Concentrations [M]')


plt.grid()


#initial Q is 52.5279, want to start at 1M. Let [LiC6] start at 52.5279, [C6] start at 1

#%%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#train_data = [files[0].iloc[40:19000, 3],files[0].iloc[40:19000, 8]]
#test_data = [files[1].iloc[40:19000, 3],files[1].iloc[40:19000, 8]]
#train_data = [files[0].iloc[40:1900, 3],files[0].iloc[40:1900, 8]]
#test_data = [files[1].iloc[40:1900, 3],files[1].iloc[40:1900, 8]]
#train_data = files[0].iloc[40:1900, 3].astype(float).to_numpy()
#train_targets = files[0].iloc[40:1900, 8].astype(float).to_numpy()

# Create dataset instances
    
#train_data = [files[0].iloc[40:1900, 3].astype(float).to_numpy(), files[0].iloc[40:1900, 8].astype(float).to_numpy()]
#test_data = files[1].iloc[40:1900, 3].astype(float).to_numpy()
#test_targets = files[1].iloc[40:1900, 8].astype(float).to_numpy()


train_data = files[0].iloc[40:1900, 3].astype(float).to_numpy().reshape(-1, 1)
train_targets = files[0].iloc[40:1900, 8].astype(float).to_numpy().reshape(-1, 1)

test_data = files[1].iloc[40:1900, 3].astype(float).to_numpy().reshape(-1, 1)
test_targets = files[1].iloc[40:1900, 8].astype(float).to_numpy().reshape(-1, 1)


train_dataset = CustomDataset(train_data, train_targets)
test_dataset = CustomDataset(test_data, test_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle=False, num_workers=0)

#IN_DIMS = (2,1896)
#IN_DIMS_SIZE = IN_DIMS[0] * IN_DIMS[1]

SEED = 42
torch.manual_seed(SEED)

IN_DIMS = 1  # Each input is a single value
OUT_DIMS_SIZE = 1  # Assuming you have 10 classes

model_linear = nn.Sequential(
    nn.Linear(1, 250),  # Input is a single value
    nn.LeakyReLU(0.1),
    nn.Dropout(p=0.4),
    nn.Linear(250, 50),
    nn.LeakyReLU(0.1),
    nn.Dropout(p=0.4),
    nn.Linear(50, OUT_DIMS_SIZE)

).to(DEVICE)

def loss_function(out, target):
    return torch.nn.MSELoss()(out, target)

# Generally, the recommended optimizer would just be Adam
# You can tune the learning rate and other options
# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

optimizer_linear = optim.Adam(model_linear.parameters(), lr=0.01)

def save_model_to_location(model, path):
    torch.save(model.state_dict(), path)
    return True

# Note that when loading, your model architecture must match identically
# to the weights set you are loading from and you must pass that model in here
def load_model_from_location(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

linear_writer = SummaryWriter(log_dir = "runs/linear_model")
model_linear.train()
epoch_count = 100

#%%

for epoch in range(epoch_count):
    # Loop through and get pairs of batch training inputs and labels
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Always start by doing zero_grad on your optimizers to reset them
        # for this batch
        optimizer_linear.zero_grad()
 
        # Convert the training batch of inputs/labels to proper device
        inputs_linear = torch.clone(inputs).to(DEVICE).unsqueeze(1)  
        labels_linear = torch.clone(labels).to(DEVICE).float()  
        # Compute the outputs for both models and compute the losses
        outputs_linear = model_linear(inputs_linear)
        loss_linear = loss_function(outputs_linear, labels_linear).float()

        # Do the backward propagation through all the calculations up
        # to this computed loss
        loss_linear.backward()
        optimizer_linear.step()

        """
        Don't want to force weights/models to be saved on your drive, so this
        is commented out. In reality, this is how you may want to approach
        recurrently saving your model every 250 iterations (in this example)
        """
        # if i%250 == 0:
        #    save_model_to_location(model_linear, "models/linear_model_" + str(int(i)))
        #    save_model_to_location(model_conv, "models/conv_model_" + str(int(i)))


        # Get the loss and plot it
        if i % 100 == 0:
            print("Linear Epoch", epoch, "Iterations", i, "loss", loss_linear.item())
            linear_writer.add_scalar('Loss/train', loss_linear.item(), i + epoch * len(train_loader))

linear_writer.close()
#%%
model_linear.eval()


test_losses = []
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE).unsqueeze(1)
        labels = labels.to(DEVICE).float()
        
        outputs = model_linear(inputs)
        loss = loss_function(outputs, labels)
        
        test_losses.append(loss.item())
        all_predictions.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(labels.cpu().numpy().flatten())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

mse = mean_squared_error(all_targets, all_predictions)

print("mse: " + mse)

#%%

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y):
    dc_t = dde.grad.jacobian(y, x, j=1)
    dc_x = dde.grad.jacobian(y, x, j=1)

    dc_xx = dde.grad.hessian(y, x, j=0)
    return (
        dc_t
        - (2/x)* dc_x
        - dc_xx
        )

def func(x): #reference solution
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

bc = dde.icbc.NeumannBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

data = dde.data.TimePDE( #time pde
    geomtime,
    pde,
    [bc, ic],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    solution=func,
    num_test=10000,
)
