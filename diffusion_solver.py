
import numpy as np
import matplotlib.pyplot as plt


M = 75
h = 1/M

Tau_base = 300
Tau = 300*np.ones((M, M))
kappa = 0.01                   #diffusion coefficient
y_bar = 0.5                     #y_bar value

#create the del^2 matrix with central differences

du= 1/h**2                       #upper diagonal value
d = (-2/h**2)                    #middle diagonal value
dl = du                         #lower diagonal value

#create the matrix in one dimension
exu = np.arange(M - 1)
exl = np.arange(M - 1) + 1

eyu = np.arange(M - 1)
eyl = np.arange(M - 1) + 1

I = np.identity(M)

#set the diagonals for the base matrices
Ax = I*d
Ax[exu, exu+1] += du
Ax[exl, exl- 1] += dl

Ay = I*d
Ay[exu, eyu+1] += du
Ay[eyl, eyl- 1] += dl

#set the neumann boundary conditions
Ax[-1, -1] = -1/h
Ax[-1, -2] = -1/h
Ay[-1,-1] = -1/h
Ay[-1, -2] = 1/h
Ay[0,0] = -1/h
Ay[0,1] = 1/h



del2_y = kappa*np.kron(I, Ay)
del2_x = kappa*np.kron(Ax, I)

del2 = (del2_y + del2_x)

LHS = del2                      #LHS array

#create the right hand side matrix and implement dirichlet boundary conditions
F = np.zeros((M, M))

#dirichlet boundary conditions
scaling_factor = -kappa/h**2
F[1:-1, 1] = 300*scaling_factor;
i = int(np.ceil(M/3));

while i <= np.floor(2*M/3):
    F[i, 1] = scaling_factor*(300 + 325*(np.sin(3*np.pi*(np.abs(i/M - y_bar))) + 1))
    i = i + 1

F_eval = F.reshape(M*M,1)

RHS = F_eval

Tau_raw = np.dot(np.linalg.inv(LHS),RHS)
Tau = Tau_raw.reshape(M, M)

plt.imshow(Tau)