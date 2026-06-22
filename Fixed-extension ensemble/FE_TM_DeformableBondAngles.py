# Description: Calculate the fixed-extension partition function and force-extension curve 
#              for a chain with deformable bond angles. 
#              The bond angle phi is normalized by pi, so the bending energy is written as
#              v_{ben}(phi) = 1/2*k_phi*(phi/pi - phi_e/180)^2. 
#              The code includes rigid bonds and extensible bonds with v_{str}(l) = 1/2*k_l*(l-l_e)^2.
#              High-precision arithmetic from mpmath is used to reduce round-off and 
#              cancellation errors in the inverse Fourier transform.              
# Authors: Jie Zhu, Laurence Brassart
# Date: June 22, 2026

import mpmath as mp
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class DBAParams:  
    kb: int
    phie: int     # equilibrium bond angle in degrees
    n_omega: int  # ω number of grids
    n_theta: int  # θ number of grids


def compute_W(theta_grid: np.ndarray, params: DBAParams):
    """
    Compute the bending kernel W(theta, theta') for deformable bond angles.

    For two neighboring bond orientations with polar angles theta and theta',
    and relative azimuthal angle omega, the bond angle phi is determined by
        cos(phi) = cos(theta) cos(theta') + sin(theta) sin(theta') cos(omega).

    The bending energy is
        V_ben(phi) = 1/2 * k_phi * (phi - phi_e)^2.

    In this code, phi is normalized by pi, so an equilibrium angle given in
    degrees is converted as
        phi_e -> phi_e / 180.

    The kernel W(theta, theta') is obtained by integrating the Boltzmann factor
    over the relative azimuthal angle omega,
        W(theta, theta') = integral_0^{2pi} exp[-V_ben(phi)] domega.

    Parameters
    ----------
    theta_grid : np.ndarray
        Uniform theta grid from 0 to pi.
    params : DBAParams
        Bending stiffness, equilibrium bond angle, and omega-grid parameters.

    Returns
    -------
    W : np.ndarray
        Bending kernel W(theta, theta') on the theta grid.
    """

    n_theta = theta_grid.size
    n_omega = params.n_omega
    domega = 2.0 * np.pi / n_omega

    # Trigonometric values on the theta grid.
    sin_t = np.sin(theta_grid)
    cos_t = np.cos(theta_grid)

    # Outer products used to evaluate
    # cos(phi) = cos(theta) cos(theta') + sin(theta) sin(theta') cos(omega)
    # for all theta-theta' pairs at each omega.
    A = np.outer(cos_t, cos_t)
    B = np.outer(sin_t, sin_t)

    W = np.zeros((n_theta, n_theta), dtype=np.float64)

    # Uniform omega quadrature without repeating the endpoint 2*pi.
    for j in range(n_omega):
        omega = domega * j
        cosw = np.cos(omega)

        cos_phi = A + B * cosw
        np.clip(cos_phi, -1.0, 1.0, out=cos_phi)

        # phi is normalized by pi. Therefore params.phie, given in degrees,
        # is converted to the same normalized unit by params.phie/180.
        phi = np.arccos(cos_phi) / np.pi
        V_ben = 0.5 * params.kb * (phi - params.phie / 180.0)**2

        W += np.exp(-V_ben)

    W *= domega

    return W


def prepare_DBA_quadrature_mp(params):
    """
    Prepare the angular quadrature and bending kernel for the deformable-bond-angle model.
    
    The bending kernel W(theta, theta') is computed using NumPy and then converted to
    mpmath objects for the high-precision Fourier-space calculation.

    Parameters
    ----------
    params : DBAParams
        Model and quadrature parameters.

    Returns
    -------
    Wmp : list of list of mp.mpf
        Bending kernel on the theta grid.
    cos_th : list of mp.mpf
        cos(theta) on the theta grid.
    sin_th : list of mp.mpf
        sin(theta) on the theta grid.
    dtheta : mp.mpf
        Theta-grid spacing.
    """

    n_theta = params.n_theta

    if n_theta < 2:
        raise ValueError("n_theta must be at least 2.")

    # Uniform theta grid including both endpoints 0 and pi.
    dtheta_np = np.pi / (n_theta - 1)
    theta_grid = dtheta_np * np.arange(n_theta)

    # Compute the angular coupling kernel W(theta, theta') in double precision.
    W = compute_W(theta_grid, params)

    # Convert theta grid and W to mpmath objects.
    dtheta = mp.pi / (mp.mpf(n_theta) - 1)
    theta_mp = [dtheta * a for a in range(n_theta)]

    cos_th = [mp.cos(theta) for theta in theta_mp]
    sin_th = [mp.sin(theta) for theta in theta_mp]

    Wmp = [
        [mp.mpf(W[a, b]) for b in range(n_theta)]
        for a in range(n_theta)
    ]

    return Wmp, cos_th, sin_th, dtheta


def inverse_fourier_qnonpos_to_zpos(Bhat_nonpos, dq, dz, nz):
    """
    Inverse Fourier transform on the nonnegative z-grid using the nonpositive
    q half-axis.

    The input Bhat_nonpos contains the Fourier-space values on

        q = -qmax, ..., -dq, 0,

    with length nz//2 + 1. The positive-q part is recovered by conjugate
    symmetry,

        Bhat(+q) = conj(Bhat(-q)).

    This convention matches the shifted Fourier grid

        q_k = dq * (k - nz//2),   k = 0, ..., nz - 1.

    Parameters
    ----------
    Bhat_nonpos : list of mp.mpf or mp.mpc
        Fourier-space values on q <= 0. The indexing is

            Bhat_nonpos[0]  -> q = -qmax,
            Bhat_nonpos[k0] -> q = 0,
            Bhat_nonpos[k0-j] -> q = -j*dq.

    dq : mp.mpf
        Fourier-grid spacing.
    dz : mp.mpf
        Extension-grid spacing.
    nz : int
        Total number of Fourier grid points. Must be even.

    Returns
    -------
    z_pos : list of mp.mpf
        Nonnegative extension grid, z_m = m*dz.
    Bz_pos : list of mp.mpc
        Inverse Fourier transform evaluated on z_pos.
    """

    if nz % 2 != 0:
        raise ValueError("nz must be even.")

    k0 = nz // 2

    if len(Bhat_nonpos) != k0 + 1:
        raise ValueError("Bhat_nonpos must have length nz//2 + 1.")

    z_pos = [dz * m for m in range(k0)]
    prefactor = dq / (2 * mp.pi)

    Bz_pos = []

    # r = exp(i*dq*z_m). Since z_m = m*dz, r can be updated recursively
    # from one z-grid point to the next.
    r_step = mp.exp(1j * dq * dz)
    r = mp.mpc(1)

    for _m in range(k0):
        # q = 0 term.
        s = Bhat_nonpos[k0]

        # Paired modes q = -j*dq and q = +j*dq.
        # The positive-q contribution is included through conjugate symmetry.
        Ej = r
        for j in range(1, k0):
            s += 2 * mp.re(Bhat_nonpos[k0 - j] * Ej)
            Ej *= r

        # Unpaired endpoint q = -qmax.
        s += Bhat_nonpos[0] * Ej

        Bz_pos.append(prefactor * s)

        # Advance from z_m to z_{m+1}.
        r *= r_step

    return z_pos, Bz_pos


def Bz_to_FE_force(Bz_pos, z_pos):
    """
    Convert the fixed-extension partition function B(z) into the FE free energy
    and the corresponding force-extension relation.

    Parameters
    ----------
    Bz_pos : list of mp.mpf or mp.mpc
        Fixed-extension partition function evaluated on the nonnegative z-grid.
    z_pos : list of mp.mpf
        Nonnegative extension grid.

    Returns
    -------
    R : np.ndarray
        Nonnegative extension grid.
    H : np.ndarray
        Dimensionless FE free energy, up to an additive constant.
    fH : np.ndarray
        Dimensionless FE force, computed as dH/dR.
    """

    H_pos = [
        -mp.log(rb) if rb > 0 else mp.nan
        for rb in (mp.re(bz) for bz in Bz_pos)
    ]

    R = np.array([float(z) for z in z_pos], dtype=np.float64)
    H = np.array([float(H) for H in H_pos], dtype=np.float64)

    # The FE force is the derivative of the FE free energy.
    fH = np.gradient(H, R, edge_order=2)

    return R, H, fH

def compute_Bhat_nonpos_DBA_rigid(N, Wmp, dq, k0, cos_th, sin_th, dtheta):
    """
    Compute the N-bond Fourier-space partition function on the nonpositive
    q half-axis for rigid bonds with deformable bond angles.

    For a rigid bond with nondimensional bond length l_e = 1, the one-bond
    Fourier factor at fixed polar angle theta is
        Qhat(q, theta) = sin(theta) exp(i q cos(theta)).

    The bending interaction is included through the transfer-matrix recursion.

    Parameters
    ----------
    N : int
        Number of bonds.
    Wmp : list of list of mp.mpf
        Bending kernel W(theta, theta').
    dq : mp.mpf
        Fourier-grid spacing.
    k0 : int
        Index of q = 0 on the shifted Fourier grid, k0 = nz//2.
    cos_th, sin_th : list of mp.mpf
        cos(theta) and sin(theta) on the theta grid.
    dtheta : mp.mpf
        Theta-grid spacing.

    Returns
    -------
    Bhat_nonpos : list of mp.mpc
        Fourier-space N-bond partition function on q <= 0.
    """

    q_start = -dq * k0
    n_theta = len(cos_th)

    # Recurrence in q for exp(i q cos(theta)).
    step = [
        mp.exp(1j * dq * cos_th[a])
        for a in range(n_theta)
    ]

    E = [
        mp.exp(1j * q_start * cos_th[a])
        for a in range(n_theta)
    ]

    Bhat_nonpos = [mp.mpc(0) for _ in range(k0 + 1)]
    WB = [mp.mpc(0) for _ in range(n_theta)]

    for k in range(k0 + 1):
        # One-bond factor at the current q value.
        Qhat = [
            E[a] * sin_th[a]
            for a in range(n_theta)
        ]

        # First bond. The factor 2*pi comes from the azimuthal integration
        # for the first bond orientation.
        B = [
            2 * mp.pi * Qhat[a]
            for a in range(n_theta)
        ]

        # Transfer-matrix recursion for the remaining bonds.
        for _ in range(1, N):
            for a in range(n_theta):
                WB[a] = dtheta * mp.fdot(Wmp[a], B)

            B = [
                Qhat[a] * WB[a]
                for a in range(n_theta)
            ]

        # Final integration over theta.
        Bhat_nonpos[k] = dtheta * mp.fsum(B)

        # Move from q_k to q_{k+1}.
        if k < k0:
            for a in range(n_theta):
                E[a] *= step[a]

    return Bhat_nonpos


def FE_DBA_rigid(N, nz, dz, params, mp_dps):
    """
    Fixed-extension force-extension relation for rigid bonds with deformable
    bond angles.

    Parameters
    ----------
    N : int
        Number of bonds.
    nz : int
        Number of Fourier grid points. Must be even.
    dz : float
        Extension-grid spacing.
    params : DBAParams
        Bending stiffness, equilibrium bond angle, and angular quadrature.
    mp_dps : int
        Decimal precision used by mpmath.

    Returns
    -------
    R : np.ndarray
        Nonnegative extension grid.
    H : np.ndarray
        Dimensionless FE free energy, up to an additive constant.
    fH : np.ndarray
        Dimensionless FE force, computed as dH/dR.
    """

    if nz % 2 != 0:
        raise ValueError("nz must be even.")

    with mp.workdps(mp_dps):
        dz = mp.mpf(str(dz))

        # The real-space computational interval is [-L, L).
        # For rigid bonds, the physical support is |R| <= N.
        L = mp.mpf(nz) * dz / 2
        if L <= N:
            raise ValueError("The real-space half-domain L = nz*dz/2 must be larger than N.")

        dq = mp.pi / L
        k0 = nz // 2

        Wmp, cos_th, sin_th, dtheta = prepare_DBA_quadrature_mp(params)

        Bhat_nonpos = compute_Bhat_nonpos_DBA_rigid(
            N=N,
            Wmp=Wmp,
            dq=dq,
            k0=k0,
            cos_th=cos_th,
            sin_th=sin_th,
            dtheta=dtheta,
        )

        # Normalize before inverse Fourier transform. This shifts H by an
        # additive constant only and therefore does not affect fH.
        Bmax = max(abs(v) for v in Bhat_nonpos)
        Bhat_nonpos = [
            v / Bmax
            for v in Bhat_nonpos
        ]

        z_pos, Bz_pos = inverse_fourier_qnonpos_to_zpos(
            Bhat_nonpos=Bhat_nonpos,
            dq=dq,
            dz=dz,
            nz=nz,
        )

        R, H, fH = Bz_to_FE_force(Bz_pos, z_pos)

    return R, H, fH


def M2_shifted_halfline_mp(alpha, l_star):
    """
    Second moment of the shifted half-line Gaussian integral.

    This function evaluates
        integral_0^infty l^2 exp[-alpha^2 (l - l_star)^2] dl,

    where alpha = sqrt(kl/2). The result is used in the extensible-bond
    one-bond factor after completing the square.

    Parameters
    ----------
    alpha : mp.mpf
        sqrt(kl/2).
    l_star : mp.mpf or mp.mpc
        Shifted Gaussian center.

    Returns
    -------
    mp.mpf or mp.mpc
        The shifted half-line second moment.
    """

    x = alpha * l_star

    return (
        mp.sqrt(mp.pi)
        * (1 + 2 * x**2)
        / (4 * alpha**3)
        * mp.erfc(-x)
        + l_star * mp.exp(-x**2) / (2 * alpha**2)
    )


def compute_Bhat_nonpos_DBA_extensible(N, Wmp, dq, k0, cos_th, sin_th, dtheta, kl):
    """
    Compute the N-bond Fourier-space partition function on the nonpositive
    q half-axis for extensible bonds with deformable bond angles.

    For an extensible bond, the bond length is integrated out analytically.
    At fixed polar angle theta, the one-bond Fourier factor contains
        exp(i q cos(theta))
        exp[-q^2 cos^2(theta)/(2 kl)]
        M2(alpha, l_star),

    where
        alpha = sqrt(kl/2),
        l_star = 1 + i q cos(theta)/kl.

    The bending interaction is included through the transfer-matrix recursion.

    Parameters
    ----------
    N : int
        Number of bonds.
    Wmp : list of list of mp.mpf
        Bending kernel W(theta, theta').
    dq : mp.mpf
        Fourier-grid spacing.
    k0 : int
        Index of q = 0 on the shifted Fourier grid, k0 = nz//2.
    cos_th, sin_th : list of mp.mpf
        cos(theta) and sin(theta) on the theta grid.
    dtheta : mp.mpf
        Theta-grid spacing.
    kl : mp.mpf
        Dimensionless bond-stretching stiffness.

    Returns
    -------
    Bhat_nonpos : list of mp.mpc
        Fourier-space N-bond partition function on q <= 0.
    """

    q_start = -dq * k0
    n_theta = len(cos_th)

    alpha = mp.sqrt(kl / 2)

    # Recurrence in q for exp(i q cos(theta)).
    step = [
        mp.exp(1j * dq * cos_th[a])
        for a in range(n_theta)
    ]

    E = [
        mp.exp(1j * q_start * cos_th[a])
        for a in range(n_theta)
    ]

    Bhat_nonpos = [mp.mpc(0) for _ in range(k0 + 1)]
    WB = [mp.mpc(0) for _ in range(n_theta)]

    q = q_start

    for k in range(k0 + 1):
        Qhat = [mp.mpc(0) for _ in range(n_theta)]

        for a in range(n_theta):
            c = cos_th[a]
            s = sin_th[a]

            l_star = 1 + 1j * q * c / kl
            M2 = M2_shifted_halfline_mp(alpha, l_star)

            Qhat[a] = (
                E[a]
                * s
                * mp.exp(-(q**2) * (c**2) / (2 * kl))
                * M2
            )

        # First bond. The factor 2*pi comes from the azimuthal integration
        # for the first bond orientation.
        B = [
            2 * mp.pi * Qhat[a]
            for a in range(n_theta)
        ]

        # Transfer-matrix recursion for the remaining bonds.
        for _ in range(1, N):
            for a in range(n_theta):
                WB[a] = dtheta * mp.fdot(Wmp[a], B)

            B = [
                Qhat[a] * WB[a]
                for a in range(n_theta)
            ]

        # Final integration over theta.
        Bhat_nonpos[k] = dtheta * mp.fsum(B)

        # Move from q_k to q_{k+1}.
        if k < k0:
            for a in range(n_theta):
                E[a] *= step[a]
            q += dq

    return Bhat_nonpos


def FE_DBA_extensible(N, nz, dz, kl, params, mp_dps):
    """
    Fixed-extension force-extension relation for extensible bonds with
    deformable bond angles.

    Parameters
    ----------
    N : int
        Number of bonds.
    nz : int
        Number of Fourier grid points. Must be even.
    dz : float
        Extension-grid spacing.
    kl : real
        Dimensionless bond-stretching stiffness.
    params : DBAParams
        Bending stiffness, equilibrium bond angle, and angular quadrature.
    mp_dps : int
        Decimal precision used by mpmath.

    Returns
    -------
    R : np.ndarray
        Nonnegative extension grid.
    H : np.ndarray
        Dimensionless FE free energy, up to an additive constant.
    fH : np.ndarray
        Dimensionless FE force, computed as dH/dR.
    """

    if nz % 2 != 0:
        raise ValueError("nz must be even.")

    with mp.workdps(mp_dps):
        dz = mp.mpf(str(dz))
        kl = mp.mpf(str(kl))

        # The real-space computational interval is [-L, L).
        # For extensible bonds, the distribution is not compactly supported,
        # so L should be chosen large enough that the tail is negligible at
        # the boundary.
        L = mp.mpf(nz) * dz / 2
        dq = mp.pi / L
        k0 = nz // 2

        Wmp, cos_th, sin_th, dtheta = prepare_DBA_quadrature_mp(params)

        Bhat_nonpos = compute_Bhat_nonpos_DBA_extensible(
            N=N,
            Wmp=Wmp,
            dq=dq,
            k0=k0,
            cos_th=cos_th,
            sin_th=sin_th,
            dtheta=dtheta,
            kl=kl,
        )

        # Normalize before inverse Fourier transform. This shifts H by an
        # additive constant only and therefore does not affect fH.
        Bmax = max(abs(v) for v in Bhat_nonpos)
        Bhat_nonpos = [
            v / Bmax
            for v in Bhat_nonpos
        ]

        z_pos, Bz_pos = inverse_fourier_qnonpos_to_zpos(
            Bhat_nonpos=Bhat_nonpos,
            dq=dq,
            dz=dz,
            nz=nz,
        )

        R, H, fH = Bz_to_FE_force(Bz_pos, z_pos)

    return R, H, fH


if __name__ == '__main__':
    
    # Bending/angular parameters shared by rigid and extensible bonds.
    params = DBAParams(
        kb=1000,
        phie=60,
        n_omega=100,
        n_theta=200,
    )

    kl = 500             # dimensionless bond-stretching stiffness
    
    # Small example for demonstration. Increase N, nz and mp_dps for production calculations.
    N = 4                # number of bonds
    nz = 1024            # number of Fourier grid points; must be even
    dz = 0.01            # extension-grid spacing

    mp_dps = 80          # decimal precision used by mpmath


    R_rigid, H_rigid, fH_rigid = FE_DBA_rigid(N, nz, dz, params, mp_dps)
    R_ext, H_ext, fH_ext = FE_DBA_extensible(N, nz, dz, kl, params, mp_dps)


    plt.figure()
    plt.rc('font',family='serif',size=14)
    plt.plot(R_rigid/N, fH_rigid, '-', linewidth=1, label='Rigid Bonds')
    plt.plot(R_ext/N, fH_ext, '-', linewidth=1, label='Extensible Bonds')
    plt.xlim([-0.05,1.25])
    plt.xticks(np.arange(0,1.25,step=0.2))
    plt.ylim([-5,95])
    plt.yticks(np.arange(0,100,step=30))
    plt.xlabel(r'$R/(Nl_e)$')
    plt.ylabel(r'$\beta fl_e$')
    plt.legend(loc='upper left')
    plt.grid(False)
    plt.show()