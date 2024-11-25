
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc


def diffusion( num_samples = 20, min_kappa = 5e-4, max_kappa = 0.025, M = 50, min_U1 = 1e-5, max_U1 = 4):
    
    print("starting...\n")
    h = 1/M
    
    num_timesteps = 100            #timesteps
    dt = 2                       #timestep size

    #create the del matrix with upwinding
    
    du = 1/h
    d = -1/h
    dl = 0
    
    #create the matrix in one dimension
    exu = np.arange(M - 1)
    exl = np.arange(M - 1) + 1
    eyu = np.arange(M - 1)
    eyl = np.arange(M - 1) + 1
    
    I = np.identity(M)
    
    Ax = I*d
    Ax[exu, exu] /= exu + 1
    Ax[exu, exu+1] += du/(exu + 1)
    Ax[exl, exl-1] += dl/(exl + 1)
    Ax[-1, -1] = -1/(h*M)
    Ax[-1, -2] = 1/(h*M)

    
        
    del1 = np.kron(I, Ax) 
    
    #create the del^2 matrix with central differences
    
    du= 1/h**2                       #upper diagonal value
    d = (-2/h**2)                    #middle diagonal value
    dl = du                         #lower diagonal value
    
    #create the matrix in one dimension
    exu = np.arange(M - 1)
    exl = np.arange(M - 1) + 1
    
    eyu = np.arange(M - 1)
    eyl = np.arange(M - 1) + 1
    

    #set the diagonals for the base matrices
    Ax = I*d
    Ax[exu, exu + 1] += du
    Ax[exl, exl- 1] += dl
    Ax[-1,-1] = 0
    Ax[-1,-2] = 0
    Ax[0,0] = 0
    Ax[0,1] = 0
    
    Ay = I*d
    Ay[eyu, eyu + 1] += du
    Ay[eyl, eyl - 1] += dl
    
    #set the neumann boundary conditions

    #Ay[-1,-1] = 0
    #Ay[-1, -2] = 0
    #Ay[0,0] = 0
    #Ay[0,1] = 0

    del2_y = np.kron(I, Ax)
    del2_x = np.kron(Ay, I)
    
    del2 = (del2_y + del2_x)
    
    #set the time stepping matrix with upwinding
    
    d = -1/dt
    
    At = I*d
    delt_x = np.kron(I, At)
    delt_y = np.kron(At, I)
    delt = delt_x + delt_y 
    
    sampler = qmc.LatinHypercube(d = 2)
    sample = sampler.random(n = 20)
    
    sample[:, 0] = (max_kappa - min_kappa)* sample[:, 0]  + min_kappa
    sample[:, 1] = (max_U1 - min_U1)* sample[:, 1]  + min_U1
    sample_idx = 0
    
    Fs = np.ones((num_samples, num_timesteps, M, M))
    F = np.zeros((M, M))
    F_prev = np.zeros((M, M))

    while sample_idx < num_samples:

        print("creating sample " + str(sample_idx) + "...\n" )
        print("kappa: ", sample[sample_idx, 0])
        print("U: ", sample[sample_idx, 1])
        #create the right hand side matrix and implement dirichlet boundary conditions

        
        #dirichlet boundary conditions
        scaling_factor = sample[sample_idx, 0]/((h**2)) - 1/(dt)
        
        i = int(np.ceil(M/3));
        
        LHS = sample[sample_idx, 0]*del2 + sample[sample_idx, 1]*del1 + delt              #LHS array
        
        t = 0
        F_prev = np.zeros((M, M))
        F_prev[0,:] = scaling_factor
        F_prev[-1,:] = 0
        
        while t < num_timesteps:
            
            print("timestep :", t)
            F_eval = F_prev.reshape(M*M,1)
            RHS = F_eval
            F = np.linalg.solve(-LHS,RHS)
            F = F.reshape(M, M)
            plt.imshow(F)
            plt.show()
            F[0, :] = scaling_factor
            F[-1,:] = 0
            F_prev = F
            Fs[sample_idx, t, :, :] = F
            t += 1
        
        sample_idx += 1
        
    
    np.save('Fs.npy', Fs)
    
    
    
diffusion()