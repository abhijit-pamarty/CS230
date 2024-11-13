
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc


def diffusion( num_samples = 20, min_ybar = 0.4, max_ybar = 0.6, min_Tau = 300, max_Tau = 500, M = 100):
    
    print("starting...\n")
    h = 1/M
    
    Tau_base = 300
    Tau = 300*np.ones((M, M))
    kappa = 0.025                   #diffusion coefficient
    y_bar = 0.4                     #y_bar value
    
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
    
    sampler = qmc.LatinHypercube(d = 2)
    sample = sampler.random(n = 20)
    
    sample[:, 0] = (max_Tau - min_Tau)* sample[:, 0]  + min_Tau
    sample[:, 1] = (max_ybar - min_ybar)* sample[:, 1]  + min_ybar
    sample_idx = 0
    
    Taus = np.ones((num_samples, M, M))
    invA = np.linalg.inv(LHS)
    
    
    while sample_idx < num_samples:

        print("creating sample " + str(sample_idx) + "...\n" )
        #create the right hand side matrix and implement dirichlet boundary conditions
        F = np.zeros((M, M))
        
        #dirichlet boundary conditions
        scaling_factor = -kappa/h**2
        F[1:-1, 1] = 300*scaling_factor;
        i = int(np.ceil(M/3));
        
        while i <= np.floor(2*M/3):
            F[i, 1] = scaling_factor*(300 + sample[sample_idx, 0]*(np.sin(3*np.pi*(np.abs(i/M - sample[sample_idx, 1]))) + 1))
            i = i + 1
        
        F_eval = F.reshape(M*M,1)
        
        RHS = F_eval
        
        Tau_raw = np.dot(invA,RHS)
        Tau = Tau_raw.reshape(M, M)
        Taus[sample_idx, :, :] = Tau
        
        sample_idx += 1
        
    
    np.save('Taus.npy', Taus)
    
    
    
diffusion()