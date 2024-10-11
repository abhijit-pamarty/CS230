# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:13:30 2024

@author: wygli
"""

#This code uses the Cottrell Equation for unidirection diffusion to estimate Lithium Ion concentrations.
# Data is taken from the DIB dataset from the University of Warwick.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path=r"C:\Users\wygli\OneDrive\Desktop\docs\Stanford Grad\cs230\DIB_Data\.csvfiles\Capacity_Check\80per_Cells_Capacity_Check_08122021_080cycle"


def getCSV(folder_path):
    files = []
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for file in csv_files:
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(folder_path, file)
        dataframe = pd.read_csv(file_path, skiprows=range(15))
        files.append(dataframe)
        return files

def Cottrell_eq(t,n,D,F,A,i):
    return i/(n*F*A*(D**(1/2))/((np.pi**(1/2))*t**(1/2)))

F = 96485 #Faraday's constant, C/mol
n = 1 #number of electrons transferred per ion
A = 1 #surface area of electrode
D = 2.5*10**(-6) #diffusion coefficient of LiPF6 in EC:DMC, cm^2/s

files = getCSV(folder_path)
concentrations = Cottrell_eq(files[0].iloc[13497:19000, 3].astype(float), n, D, F, A, files[0].iloc[13497:19000, 8].astype(float))

plt.ylabel('Concentrations')
plt.plot(concentrations)
plt.grid()