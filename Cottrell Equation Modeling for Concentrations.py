# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:13:30 2024

@author: wygli
"""

# This code uses the Cottrell Equation for unidirection diffusion to estimate Lithium Ion concentrations.
# Data is taken from the DIB dataset from the University of Warwick.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

F = 96485 #Faraday's constant, C/mol
n = 1 #number of electrons transferred per ion
A = 1 #surface area of electrode
D = 2.5*10**(-6) #diffusion coefficient of LiPF6 in EC:DMC, cm^2/s

folder_path=r"C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\DIB_Data\.csvfiles\Capacity_Check\80per_Cells_Capacity_Check_08122021_080cycle"
files = getCSV(folder_path)
concentrations = Cottrell_eq(files[0].iloc[40:19000, 3].astype(float), n, D, F, A, abs(files[0].iloc[40:19000, 8].astype(float)))

plt.ylabel('Concentrations')
plt.plot(concentrations)
plt.grid()
