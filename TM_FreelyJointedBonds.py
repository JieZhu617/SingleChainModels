# Description: Calculate the bond orientation probabilities and force-extension curve in a chain with freely jointed bonds
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np
import matplotlib.pyplot as plt


def Langevin(x):
    # Calculate the Lagevin function: y = coth(x)-1/x 
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


def FJB_Rigid(f,Mtheta):
    # Rigid bonds
    # Input: f--Applied force; Mtheta--Mesh parameter
    # Output: P--Bond orientation probability; G--Gibbs free energy; r--Chain end-to-end distance
    theta = np.linspace(0,1,Mtheta)
    cos_theta = np.cos(np.pi*theta)

    if not isinstance(f,np.ndarray):
        W = np.exp(f*cos_theta)*np.sin(np.pi*theta)
        P = W/np.trapz(W,theta)
        if f == 0:
            G = -np.log(2/np.pi)
            r = 0
        else:
            G = -np.log(np.sinh(f)/f) - np.log(2/np.pi)
            r = Langevin(f)
    else:
        W = np.exp(f[:,np.newaxis]*cos_theta)*np.sin(np.pi*theta)
        P = W/np.trapz(W,theta,axis=1)[:,np.newaxis]

        G = np.zeros(len(f)) - np.log(2/np.pi)
        r = np.zeros(len(f))
        G[f!=0] = -np.log(np.sinh(f[f!=0])/f[f!=0]) - np.log(2/np.pi)
        r[f!=0] = Langevin(f[f!=0])
    
    return P,G,r

def FJB_Extensible(f,k_l,l_min,l_max,Mtheta,Ml):
    # Extensible bonds with harmonic stretching energy: v_{str}(l) = 1/2*k_l*(l-1)^2
    # Input: f--Applied force; Mtheta--Mesh parameter of theta;
    #        k_l--Normalized bond stretching stiffness (=k_l*l_e/k_BT);
    #        l_min--Normalized minimum bond length (=l_min/l_e);
    #        l_max--Normalized maximum bond length (=l_min/l_e);
    #        Ml--Mesha paramter of l;
    # Output: P--Bond orientation probability; G--Gibbs free energy; r--Chain end-to-end distance 
    theta = np.linspace(0,1,Mtheta)
    l = np.linspace(l_min,l_max,Ml)
    cos_theta = np.cos(np.pi*theta)

    if not isinstance(f,np.ndarray):
        extforce = f*l[:,np.newaxis]*np.cos(np.pi*theta)
        extension = -k_l/2*(l[:,np.newaxis]-1)**2
        W_fun = np.exp(extension + extforce)*l[:,np.newaxis]**2*np.sin(np.pi*theta)
        W = np.trapz(W_fun,l,axis=0)
        P = W/np.trapz(W,theta)

        if f == 0:
            G_fun = np.exp(-k_l/2*(l-1)**2)*l**2
            G = -np.log(np.trapz(G_fun,l)) - np.log(2/np.pi)
            r = 0
        else:
            G_fun = np.exp(-k_l/2*(l-1)**2)*(np.sinh(f*l)/f)*l
            G = -np.log(np.trapz(G_fun,l,axis=0)) - np.log(2/np.pi)

            r_fun1 = np.exp(-k_l/2*(l-1)**2)*np.cosh(f*l)*l**2
            r_fun2 = np.exp(-k_l/2*(l-1)**2)*np.sinh(f*l)*l
            r = np.trapz(r_fun1,l)/np.trapz(r_fun2,l) - 1/f
    else:
        extforce = f[:,np.newaxis,np.newaxis]*l[:,np.newaxis]*cos_theta
        extension = -k_l/2*(l[:,np.newaxis]-1)**2
        W_fun = np.exp(extension + extforce)*l[:,np.newaxis]**2*np.sin(np.pi*theta)
        W = np.trapz(W_fun,l,axis=1)
        P = W/np.trapz(W,theta,axis=1)[:,np.newaxis]

        G = np.zeros(len(f))
        G_fun = np.zeros([len(f),len(l)])
        G_fun[f==0] = np.exp(-k_l/2*(l-1)**2)*l**2
        G_fun[f!=0] = np.exp(-k_l/2*(l-1)**2)*(np.sinh(f[:,np.newaxis][f!=0]*l)/f[:,np.newaxis][f!=0])*l
        G = -np.log(np.trapz(G_fun,l,axis=1)) - np.log(2/np.pi)

        r = np.zeros(len(f))
        r_fun1 = np.zeros([len(f),len(l)])
        r_fun2 = np.zeros([len(f),len(l)])
        r_fun1[f!=0] = np.exp(-k_l/2*(l-1)**2)*np.cosh(f[:,np.newaxis][f!=0]*l)*l**2
        r_fun2[f!=0] = np.exp(-k_l/2*(l-1)**2)*np.sinh(f[:,np.newaxis][f!=0]*l)*l
        r[f!=0] = np.trapz(r_fun1[f!=0],l,axis=1)/np.trapz(r_fun2[f!=0],l,axis=1) - 1/f[f!=0]
    
    return P,G,r


if __name__ == '__main__':

    k_l = 1000
    l_min = 0
    l_max = 2
    Mtheta = 200
    Ml = 100
    theta = np.linspace(0,np.pi,Mtheta)
    f = np.linspace(0.1,100,100)
    # f = 10

    Pr,Gr,rr = FJB_Rigid(f,Mtheta)
    P,G,r = FJB_Extensible(f,k_l,l_min,l_max,Mtheta,Ml)


    # Plot the force-extension curves
    plt.figure()
    plt.rc('font',family='serif',size=14)
    plt.plot(f,rr,'-', linewidth=2,label='Rigid Bonds')
    plt.plot(f,r,'--', linewidth=2,label='Extensible Bonds')
    plt.xlim([-1,101])
    plt.ylim([-0.02,1.22])
    plt.xticks(np.arange(0,101,step=20))
    plt.yticks(np.arange(0,1.21,step=0.2))
    plt.xlabel(r'$fl_e/k_BT$')
    plt.ylabel(r'$r/Nl$')
    plt.legend(loc='lower right')
    plt.grid('false')
    plt.show()