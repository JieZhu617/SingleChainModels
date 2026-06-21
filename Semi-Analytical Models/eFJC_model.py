# Description: Calculate the force-extension curve using the eFJC model
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Jedynak approximation of the inverse Langevin function
def invLangevinJedynak(x):
    err = 1e-10
    if not isinstance(x,np.ndarray):
        if x >= 1-err:
            x = 1-err
    else:
        x = np.clip(x, None, 1-err)
    return x * (3-2.6*x+0.7*x**2) / ((1-x)*(1+0.1*x))

# Potential of Extention
def PotentialExt(l,k_l):
    return k_l*(l-1.0)**2/2

# Potential of Bending
def PotentialBen(phi,phi_e,k_phi):
    return k_phi*(phi-phi_e)**2/2

# Entropic free energy of FJC
def EntropicFE_FJC(r,l):
    r_star = r/l

    if r_star > 1-1e-10:
        r_star = 1-1e-10
    beta = invLangevinJedynak(r_star)

    if beta > 20:
        log_sinh_beta = beta - np.log(2)
    else:
        log_sinh_beta = np.log(np.sinh(beta))

    return r_star*beta + np.log(beta) - log_sinh_beta


# Constitutive Relation of FJC
def ConRel_FJC(r,l):

    r_star = r/l
    beta = invLangevinJedynak(r_star)/l

    return beta

# eFJC model
def FJC_ExtensibleBonds(r,k_l,l_min,l_max):
    bound_l = [(l_min+1e-6,l_max)]
    Nr = len(r)
    l = np.zeros(Nr)
    l_guess = 1.0

    for i in range(Nr):
        fun = lambda x: PotentialExt(x,k_l) + EntropicFE_FJC(r[i],x)
        result = minimize(fun, l_guess, method='L-BFGS-B', bounds=bound_l, constraints=None, tol=1e-6)
        l[i] = result.x[0]
        l_guess = l[i]

    f = ConRel_FJC(r,l)
    return f,l


if __name__ == '__main__':

    k_l = 1000
    l_min = 0
    l_max = 2
    r1 = 10**np.linspace(-3,-1,200+1)
    r2 = np.linspace(0.1,1.1,500+1)
    r = np.concatenate((r1[0:len(r1)-1], r2))
    
    fr = ConRel_FJC(r,1.0)
    f,l = FJC_ExtensibleBonds(r,k_l,l_min,l_max)


    # Plot the force-extension curves
    plt.figure()
    plt.rc('font',family='serif',size=14)
    plt.plot(fr,r,'-', linewidth=2,label='Rigid Bonds')
    plt.plot(f,r,'--', linewidth=2,label='Extensible Bonds')
    plt.xlim([-1,101])
    plt.ylim([-0.02,1.22])
    plt.xticks(np.arange(0,101,step=20))
    plt.yticks(np.arange(0,1.21,step=0.2))
    plt.xlabel(r'$fl_e/k_BT$')
    plt.ylabel(r'$r/Nl_e$')
    plt.legend(loc='lower right')
    plt.grid('false')
    plt.show()

