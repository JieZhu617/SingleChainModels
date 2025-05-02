# Description: Calculate the bond orientation probabilities and force-extension curve in a chain with deformable bond angles
#              with harmonic bending energy: v_{ben}(l) = 1/2*k_phi*(phi-phi_e)^2
# Authors: Jie Zhu, Laurence Brassart
# Date: May 02, 2025

import numpy as np
import matplotlib.pyplot as plt


def BondAngle(theta_new,theta,omega):
    # Calculate bond angle
    # Input: theta_new--polar angle of next bond direction; theta--polar angle of bond direction;
    #        omega--difference in azimuthal angle between two adjacent bonds
    # Output: Bond angle
    cos_phi = np.sin(np.pi*theta_new)*np.sin(np.pi*theta)*np.cos(2*np.pi*omega) + np.cos(np.pi*theta_new)*np.cos(np.pi*theta)
    np.clip(cos_phi,-1,1,out=cos_phi)
    return np.arccos(cos_phi)/np.pi


def NextW_DBA_Rigid(f,P,Momega,k_phi,phi_e,theta):
    # Rigid bonds
    # Input: f--Applied force; P--Bond orientation probability of last bond
    #        k_phi--Normalized bond bending stiffness (=k_phi*pi^2/k_BT);
    #        phi_e--Equilibrium bond angle; theta--polar angle of bond direction;
    # Output: Weighted orientation probability of next bond
    omega = np.linspace(0,1,Momega+1)
    omega = omega[0:-1]
    phi = BondAngle(theta[:,np.newaxis,np.newaxis],theta[:,np.newaxis],omega)

    extforce = f*np.cos(np.pi*theta)
    bending = -k_phi/2*(phi - phi_e)**2
    
    T1 = np.trapz(np.exp(bending),omega,axis=2)
    W_new = np.trapz(P*T1,theta,axis=1)*np.exp(extforce)*np.sin(np.pi*theta)
 
    return W_new


def DBA_Rigid(f,N,Mtheta,Momega,k_phi,phi_e):
    # Rigid bonds
    # Input: f--Applied force; N--Number of bonds in the chain;
    #        Mtheta--Mesh parameter of theta; Momega--Mesha parameter of omega;
    #        k_phi--Normalized bond bending stiffness (=k_phi*pi^2/k_BT);
    #        phi_e--Equilibrium bond angle;
    # Output: P--Orientation probabilities P_i(\theta) of each bond
    #         G--Gibbs free energy
    theta = np.linspace(0,1,Mtheta)

    W1 = np.exp(f*np.cos(np.pi*theta))*np.sin(np.pi*theta)*2*np.pi
    intW1 = np.trapz(W1,theta)
    P1 = W1/intW1

    W = np.zeros([N,Mtheta])
    intW = np.zeros(N)
    P = np.zeros([N,Mtheta])

    W[0,:] = W1
    intW[0] = intW1
    P[0,:] = P1

    for i in range(1,N):
        W[i,:] = NextW_DBA_Rigid(f,P[i-1,:],Momega,k_phi,phi_e,theta)
        intW[i] = np.trapz(W[i,:],theta)

        P[i,:] = W[i,:]/intW[i]
    
    G = -np.sum(np.log(intW))/N

    return P,G


def NextW_DBA_Extensible(f,P,Momega,k_l,l_min,l_max,Ml,k_phi,phi_e,theta):
    # Extensible bonds with harmonic stretching energy: v_{str}(l) = 1/2*k_l*(l-1)^2
    # Input: f--Applied force; P--Bond orientation probability of last bond
    #        Momega--Mesh parameter of omega; Ml--Mesha paramter of l;
    #        k_l--Normalized bond stretching stiffness (=k_l*l_e/k_BT);
    #        l_min--Normalized minimum bond length (=l_min/l_e);
    #        l_max--Normalized maximum bond length (=l_min/l_e);
    #        k_phi--Normalized bond bending stiffness (=k_phi*pi^2/k_BT);
    #        phi_e--Equilibrium bond angle; theta--Polar angle of bond direction;       
    # Output: Weighted orientation probability of next bond
    omega = np.linspace(0,1,Momega+1)
    omega = omega[0:-1]
    l = np.linspace(l_min,l_max,Ml)
    phi = BondAngle(theta[:,np.newaxis,np.newaxis],theta[:,np.newaxis],omega)   

    extforce = f*l*np.cos(np.pi*theta[:,np.newaxis])
    extension = -k_l/2*(l-1.0)**2
    bending = -k_phi/2*(phi - phi_e)**2

    T1 = np.trapz(np.exp(bending),omega,axis=2)
    T2 = np.trapz(np.exp(extension + extforce)*l**2,l,axis=1)*np.sin(np.pi*theta)
    W_new = np.trapz(P*T1,theta,axis=1)*T2

    return W_new


def DBA_Extensible(f,N,Mtheta,Momega,k_l,l_min,l_max,Ml,k_phi,phi_e):
    # Extensible bonds with harmonic stretching energy: v_{str}(l) = 1/2*k_l*(l-1)^2
    # Input: f--Applied force; N--Number of bonds in the chain
    #        Mtheta--Mesh parameter of theta; Momega--Mesh parameter of omega;
    #        k_l--Normalized bond stretching stiffness (=k_l*l_e/k_BT);
    #        l_min--Normalized minimum bond length (=l_min/l_e);
    #        l_max--Normalized maximum bond length (=l_min/l_e);
    #        Ml--Mesha paramter of l;
    #        k_phi--Normalized bond bending stiffness (=k_phi*pi^2/k_BT);
    #        phi_e--Equilibrium bond angle;  
    # Output: P--Orientation probabilities P_i(\theta) of each bond
    #         G--Gibbs free energy
    theta = np.linspace(0,1,Mtheta)
    l = np.linspace(l_min,l_max,Momega)

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

    for i in range(1,N):
        W[i,:] = NextW_DBA_Extensible(f,P[i-1],Momega,k_l,l_min,l_max,Ml,k_phi,phi_e,theta)
        intW[i] = np.trapz(W[i,:],theta)
        P[i,:] = W[i,:]/intW[i]
    
    G = -np.sum(np.log(intW))/N

    return P,G



if __name__ == '__main__':

    k_phi = 1000
    phi_e = 60/180

    Mtheta = 200
    theta = np.linspace(0,np.pi,Mtheta)
    Momega = 20

    k_l = 1000
    l_min = 0
    l_max = 2
    Ml = 20
   
    f = np.linspace(0.1,100,100)
    N = 100

    
    Pr = np.zeros([len(f),N,Mtheta])
    Gr = np.zeros(len(f))
    P = np.zeros([len(f),N,Mtheta])
    G = np.zeros(len(f))
    for i in range(len(f)):
        Pr[i,:,:],Gr[i] = DBA_Rigid(f[i],N,Mtheta,Momega,k_phi,phi_e)
        P[i,:,:],G[i] = DBA_Extensible(f[i],N,Mtheta,Momega,k_l,l_min,l_max,Ml,k_phi,phi_e)

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