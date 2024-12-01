# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:54:29 2024

@author: wygli
"""

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
print("NumPy version:", np.__version__)
# Backend pytorch
import torch
import numpy as np
import datetime, os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
folder_path=r"C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\DIB_Data\.csvfiles\Capacity_Check\80per_Cells_Capacity_Check_08122021_080cycle"
files = getCSV(folder_path)

n = 1 #number of electrons transferred per ion
A = 1 #surface area of electrode
D = 2.5*10**(-6) #diffusion coefficient of LiPF6 in EC:DMC, cm^2/s
gas_const = 8.314 #gas constant
T = 25+273.15 #temperature
E0 = 3.7
# Problem parameter
c0 = 1.0 # initial concentration
D = 1.5*10**(-11) # solid-phase diffusion coefficient
R = 1.0
t = 1.0 # time 
It = 1.0 # current at time t
A = 1.0 # battery sheet area
L = 1.0 # thickness of positive/negative electrode
eps = 0.5 # solid phase volume fraction of each electrode
a = 3*eps/R # specific interfacial area
F = 96485 #Faraday's constant, C/mol
delta = It*R/(A*L*F*D*a*c0)

# Calculating total concentration in electrode using Nernst equation



plt.ylabel('Concentrations [M]')
plt.grid()

Q=[]
li_conc=[]
time = []
for i in range(len(files)):
    Q.append(Nernst_eq(files[i].iloc[40:19000,7].astype(float), E0, n, F, R,T))
    lic6_conc = Q[i].iloc[0]
    c6_conc = 1
    li_conc.append(Q[i]*c6_conc/lic6_conc)
    time.append(files[i].iloc[40:19000,3].astype(float))

combined = list(zip(time, li_conc))
combined0 = np.array(combined[0])

time = np.array(time)
li_conc = np.array(li_conc)

time0 = time[0]
time0 = time0[:15000]
time0=time0[::10]
li_conc0 = li_conc[0]
li_conc0 = li_conc0[:15000]
li_conc0=li_conc0[::10]

tau0 = D * time0 / (R**2)
tau_max = np.max(tau0)

#%%
# xt = (x, t)

fs_test = np.load(r'C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\Fs_test.npy')
sample_data_test = np.load(r'C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\sample_data_test.npy')
scaling_factors_test = np.load(r'C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\scaling_factors_test.npy')

scaled_1=fs_test[0]/scaling_factors_test[0]
tt, xx, yy = scaled_1.shape
x = np.linspace(0, 1, xx)
t = np.linspace(0, 1, tt)
X, T = np.meshgrid(x, t, indexing='ij')
data_xt = np.column_stack((X.flatten(), T.flatten()))
conc_xt = scaled_1[:,:,1].flatten()

def pde(xt, c):
    # Most backends
    dc_x = dde.grad.jacobian(c, xt, j=0)
    dc_t = dde.grad.jacobian(c, xt, j=1)
    dc_xx = dde.grad.hessian(c, xt, j=0)
    # Backend pytorch
    return (
        dc_t
        - 2 * dc_x / (xt[:, 0] + 1e-6)
        - dc_xx
    )

def boundary_l(xt, on_boundary):
    return on_boundary and dde.utils.isclose(xt[0], 0, atol=0.01)

def boundary_r(xt, on_boundary):
    return on_boundary and dde.utils.isclose(xt[0], 1, atol=0.01)

def initial_condition(xt):
    return np.isclose(xt[:, 0], 0,atol=0.01).astype(np.float32)

def solution_func(x):
    return conc_xt[0:1000]

# Concentration from dataset
#n_obs = time0.shape
#observe_xt = np.vstack((np.ones(n_obs), np.array(time0))).T
#observe_c = li_conc0.reshape(-1,1)
observe_pts = dde.icbc.PointSetBC(data_xt[0:1000], conc_xt[0:1000], component=0)
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#bc_l = dde.icbc.NeumannBC(geomtime, lambda X: 0.0, boundary_l)
#bc_r = dde.icbc.NeumannBC(geomtime, lambda X: 1, boundary_r)
bc_l = dde.icbc.DirichletBC(geomtime, lambda x: 1, boundary_l)
bc_r = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_r)

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, ic, observe_pts],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=data_xt[0:1000],
    solution=solution_func,
    num_test=100,
)

layer_size = [2] + [64] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory_adam, train_state_adam = model.train(iterations=3000)

model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train(iterations=10000)


# Save and plot the results
dde.saveplot(losshistory_adam, train_state_adam, issave=True, isplot=True)
dde.saveplot(losshistory_lbfgs, train_state_lbfgs, issave=True, isplot=True)

#%%

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, T = np.meshgrid(x, t)
X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]
points = np.hstack((X_flat, T_flat))

c_pred = model.predict(points)
C_pred = c_pred.reshape(X.shape)

C_pred_2d = np.expand_dims(C_pred, axis=1)
np.shape(C_pred_2d)

C_pred_2d = np.repeat(C_pred_2d, 100, axis=1)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, C_pred)
ax.set_xlabel('Position X')
ax.set_ylabel('Time T')
ax.set_zlabel('Concentration')
ax.set_title('Lithium Ion Battery Concentration')
plt.show()

#%%
C_pred_2d_fixT = C_pred_2d[:,:,90]
plt.figure(figsize=(10, 8))
plt.imshow(C_pred_2d_fixT, cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('X position')
plt.ylabel('Y position')

plt.show()
