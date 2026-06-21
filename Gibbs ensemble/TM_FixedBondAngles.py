# Description: Calculate the bond orientation probabilities and force-extension curve in a chain with fixed bond angles
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def Matrix_T(phi_e, theta):
    # Calculate Transfer Matrix (TM) operator
    # Input: phi_e--Equilibrium bond angle; theta--polar angle of bond direction
    # Output: T--TM operator; theta_p--Roots of omega
    cos_phi_e = np.cos(phi_e * np.pi)
    sin_phi_e = np.sin(phi_e * np.pi)
    sin_theta = np.sin(np.pi * theta)
    cos_theta = np.cos(np.pi * theta)

    Mtheta = len(theta)
    root1, root2 = phi_e, phi_e
    T = np.zeros([Mtheta,Mtheta])
    theta_p = np.zeros([Mtheta,Mtheta])
    for i in range(Mtheta-2):
        func = lambda x: np.abs(sin_theta[i+1]*np.sin(np.pi * x))-np.abs(cos_phi_e-cos_theta[i+1]*np.cos(np.pi * x))
        if i<round(Mtheta*phi_e-1):
            x01 = root1-1.0/Mtheta
        else:
            x01 = root1+1.0/Mtheta
        if i <round(Mtheta*(1-phi_e)-1):
            x02 = root2+1.0/Mtheta
        else:
            x02 = root2-1.0/Mtheta
        root1 = fsolve(func, x0=x01, xtol=1e-6)[0]
        root2 = fsolve(func, x0=x02, xtol=1e-6)[0]
        theta_p[i+1] = np.linspace(np.maximum(root1,0.0), np.minimum(root2,1.0), Mtheta)
        denominator = (sin_theta[i+1]*np.sin(np.pi * theta_p[i+1]))**2-(cos_phi_e-cos_theta[i+1]*np.cos(np.pi * theta_p[i+1]))**2
        T[i + 1, :] = np.where(
            denominator < 1e-12,
            0,  # Set T to zero if denominator is invalid
            2.0 * sin_phi_e / np.sqrt(np.maximum(denominator, 1e-12))
        )
    return T, theta_p


def NextW_FBA_Rigid(f, P, phi_e, theta, T, theta_p):
    # Rigid bonds
    # Input: f--Applied force; P--Bond orientation probability of last bond
    #        phi_e--Equilibrium bond angle; theta--polar angle of bond direction;
    #        T--TM operator; theta_p--Roots of omega
    # Output: Weighted orientation probability of next bond
    cos_theta = np.cos(np.pi * theta)
    extforce = f * cos_theta

    Mtheta = len(theta)
    W_new = np.zeros(Mtheta)
    P_interp = interp1d(theta,P,kind='quadratic')
    for i in range(Mtheta-2):
        W_new[i+1] = np.trapz(P_interp(theta_p[i+1])*T[i+1,:],theta_p[i+1])
    W_new[0] = P_interp(phi_e)
    W_new[-1] = P_interp(1-phi_e)

    return W_new*np.exp(extforce)*np.sin(np.pi*theta)


def FBA_Rigid(f,N,Mtheta,phi_e):
    # Rigid bonds
    # Input: f--Applied force; N--Number of bonds in the chain
    #        Mtheta--Mesh parameter; phi_e--Equilibrium bond angle;
    # Output: P--Orientation probabilities P_i(\theta) of each bond
    #         G--Gibbs free energy
    theta = np.linspace(0,1,Mtheta)

    W1 = np.exp(f*np.cos(np.pi*theta))*np.sin(np.pi*theta)
    intW1 = np.trapz(W1,theta)
    P1 = W1/intW1

    W = np.zeros([N,Mtheta])
    intW = np.zeros(N)
    P = np.zeros([N,Mtheta])

    W[0,:] = W1
    intW[0] = intW1
    P[0,:] = P1

    T,theta_p = Matrix_T(phi_e, theta)

    for i in range(1,N):
        W[i,:] = NextW_FBA_Rigid(f,P[i-1,:],phi_e,theta, T, theta_p)
        intW[i] = np.trapz(W[i,:],theta)
        P[i,] = W[i,]/intW[i]
    
    G = -np.sum(np.log(intW))/N

    return P,G


def NextW_FBA_Extensible(f, P, phi_e, theta, T, theta_p, k_l, l_max, l_min, Ml):
    # Extensible bonds with harmonic stretching energy: v_{str}(l) = 1/2*k_l*(l-1)^2
    # Input: f--Applied force; P--Bond orientation probability of last bond
    #        phi_e--Equilibrium bond angle; theta--polar angle of bond direction;
    #        T--TM operator; theta_p--Roots of omega;
    #        k_l--Normalized bond stretching stiffness (=k_l*l_e/k_BT);
    #        l_min--Normalized minimum bond length (=l_min/l_e);
    #        l_max--Normalized maximum bond length (=l_min/l_e);
    #        Ml--Mesha paramter of l;
    # Output: Weighted orientation probability of next bond
    l = np.linspace(l_min,l_max,Ml)
    extforce = f*l*np.cos(np.pi*theta[:,np.newaxis])
    extension = -k_l/2*(l-1.0)**2

    Mtheta = len(theta)
    W_new = np.zeros(Mtheta)
    P_interp = interp1d(theta,P,kind='quadratic')
    for i in range(Mtheta-2):
        W_new[i+1] = np.trapz(P_interp(theta_p[i+1])*T[i+1,:],theta_p[i+1])
    W_new[0] = P_interp(phi_e)
    W_new[-1] = P_interp(1-phi_e)

    return W_new*np.trapz(np.exp(extension + extforce)*l**2,l,axis=1)*np.sin(np.pi*theta)


def FBA_Extensible(f, N, Mtheta, phi_e, k_l, l_max, l_min, Ml):
    # Extensible bonds with harmonic stretching energy: v_{str}(l) = 1/2*k_l*(l-1)^2
    # Input: f--Applied force; N--Number of bonds in the chain
    #        Mtheta--Mesh parameter of theta; phi_e--Equilibrium bond angle;
    #        k_l--Normalized bond stretching stiffness (=k_l*l_e/k_BT);
    #        l_min--Normalized minimum bond length (=l_min/l_e);
    #        l_max--Normalized maximum bond length (=l_min/l_e);
    #        Ml--Mesha paramter of l;
    # Output: P--Orientation probabilities P_i(\theta) of each bond
    #         G--Gibbs free energy
    theta = np.linspace(0,1,Mtheta)
    l = np.linspace(l_min,l_max,Ml)

    extforce = f*l*np.cos(np.pi*theta[:,np.newaxis])
    extension = -k_l/2*(l-1.0)**2

    W1 = np.trapz(np.exp(extension + extforce)*l**2,l,axis=1)*np.sin(np.pi*theta)
    intW1 = np.trapz(W1,theta)
    P1 = W1/intW1

    W = np.zeros([N,Mtheta])
    intW = np.zeros(N)
    P = np.zeros([N,Mtheta])

    W[0,:] = W1
    intW[0] = intW1
    P[0,:] = P1

    T,theta_p = Matrix_T(phi_e, theta)

    for i in range(1,N):
        W[i,:] = NextW_FBA_Extensible(f, P[i-1,:], phi_e, theta, T, theta_p, k_l, l_max, l_min, Ml)
        intW[i] = np.trapz(W[i,:],theta)
        P[i,] = W[i,]/intW[i]
    
    G = -np.sum(np.log(intW))/N

    return P,G


if __name__ == '__main__':

    k_l = 1000
    l_min = 0
    l_max = 2
    Ml = 100

    Mtheta = 400
    theta = np.linspace(0,np.pi,Mtheta)

    f = np.linspace(0.1,100,100)
    N = 100
    phi_e = 60/180
    
    Pr = np.zeros([len(f),N,Mtheta])
    Gr = np.zeros(len(f))
    P = np.zeros([len(f),N,Mtheta])
    G = np.zeros(len(f))
    for i in range(len(f)):
        Pr[i,:,:],Gr[i] = FBA_Rigid(f[i],N,Mtheta,phi_e)
        P[i,:,:],G[i] = FBA_Extensible(f[i], N, Mtheta, phi_e, k_l, l_max, l_min, Ml)

    rr = -np.gradient(Gr,f)
    r = -np.gradient(G,f)
    
    
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
