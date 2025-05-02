# Description: Calculate the Lagevin function: y = coth(x)-1/x 
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np

def Langevin(x):
    if not isinstance(x,np.ndarray):
        return coth(x) - 1/x
    else:
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = coth(x[i]) - 1/x[i]
        return y
        

def coth(x):
    if abs(x) < 1e-15:  # Avoid division by zero
        return np.inf if x > 0 else -np.inf
    elif abs(x) > 710:  # Limit for avoiding overflow
        return 1.0 / np.tanh(x)
    else:
        return np.cosh(x) / np.sinh(x)