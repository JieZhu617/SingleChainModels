# Description: Calculate the force-extension curve using the dFRC model
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt

# Incomplete beta function
def incBetaFun(x,a,b):
    fun = lambda t: t**(a-1)*(1-t)**(b-1)
    if x <= 1-1e-4:
        y,err = integrate.quad(fun,0,x,limit=100)
        return y
    elif x <= 1-1e-8:
        y1,err1 = integrate.quad(fun,0,1-1e-4,limit=100)
        y2,err2 = integrate.quad(fun,1-1e-4,x,limit=100)
        return y1+y2
    else:
        y1,err1 = integrate.quad(fun,0,1-1e-4,limit=100)
        y2,err2 = integrate.quad(fun,1-1e-4,1-1e-8,limit=100)
        y3,err3 = integrate.quad(fun,1-1e-8,x,limit=100)
        return y1+y2+y3


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


# Entropic free energy
def EntropicFE(r,l,phi):
    R_max = l*np.cos(np.pi*phi/2)
    l_k = l*2*np.cos(np.pi*phi/2)/(1-np.cos(np.pi*phi))
    r_star = r/R_max

    if r_star > 1-1e-10:
        r_star = 1-1e-10

    beta = invLangevinJedynak(r_star)
    B = incBetaFun(r_star,2+l_k/l,-1)
    # print(r_star,B)

    if beta > 20:
        log_sinh_beta = beta - np.log(2)
    else:
        log_sinh_beta = np.log(np.sinh(beta))

    part1 = r_star*beta + np.log(beta) - log_sinh_beta
    part2 = 1/(1-r_star) + r_star + 2*np.log(1-r_star) - B
    return (1-np.cos(np.pi*phi))/4*(2*part1 + part2)


# Constitutive Relation of FRC (Eq.8 in the manuscript)
def ConRel(r,l,phi):

    R_max = l*np.cos(np.pi*phi/2)
    l_k = l*2*np.cos(np.pi*phi/2)/(1-np.cos(np.pi*phi))
    r_star = r/R_max

    beta = invLangevinJedynak(r_star)

    f = (beta + 1/2*r_star**2/(1-r_star)**2*(1-r_star**(l_k/l-1)))/l_k
    
    return f

# dFRC model
def FRC_DeformableBonds(r,k_l,l_min,l_max,k_phi,phi_e,N):
    Nr = len(r)
    l = np.zeros(Nr)
    phi = np.zeros(Nr)
    initial_guess = np.array([1.0, phi_e])
    bounds = [(l_min+1e-6,l_max),(1e-6,1)]

    for i in range(Nr):
        fun = lambda vars: PotentialExt(vars[0],k_l) + (N-1)/N*PotentialBen(vars[1],phi_e,k_phi) + EntropicFE(r[i],vars[0],vars[1])
        result = minimize(fun, initial_guess, method='L-BFGS-B', bounds=bounds, constraints=None, tol=1e-6)
        l[i] = result.x[0]
        phi[i] = result.x[1]
        initial_guess = np.array([l[i], phi[i]])

    f = ConRel(r,l,phi)
    return f,l,phi


if __name__ == '__main__':

    k_l = 1000
    l_min = 0
    l_max = 2

    phi_e = 60/180
    k_phi = 1000

    N = 100

    r1 = 10**np.linspace(-3,-1,200+1)
    r2 = np.linspace(0.1,1.1,500+1)
    r = np.concatenate((r1[0:len(r1)-1], r2))
    
    fr = ConRel(r,1.0,phi_e)
    f,l,phi = FRC_DeformableBonds(r,k_l,l_min,l_max,k_phi,phi_e,N)


    # Plot the force-extension curves
    plt.figure()
    plt.rc('font',family='serif',size=14)
    plt.plot(fr,r,'-', linewidth=2,label='Rigid Bonds & Fixed Bond Angles')
    plt.plot(f,r,'--', linewidth=2,label='Deformable Bonds')
    plt.xlim([-1,101])
    plt.ylim([-0.02,1.22])
    plt.xticks(np.arange(0,101,step=20))
    plt.yticks(np.arange(0,1.21,step=0.2))
    plt.xlabel(r'$fl_e/k_BT$')
    plt.ylabel(r'$r/Nl_e$') 
    plt.legend(loc='lower right')
    plt.grid('false')
    plt.show()