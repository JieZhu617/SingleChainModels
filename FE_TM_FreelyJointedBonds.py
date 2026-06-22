# Description: Calculate the fixed-extension partition function and force-extension curve in a chain with freely jointed bonds
#              including rigid bonds and extensible bonds (v_{str}(l) = 1/2*k_l*(l-l_e)^2)
# Authors: Jie Zhu, Laurence Brassart
# Date: June 22, 2026

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt


def Qbar_extensible_mp(s, kl):
    """
    Angle-integrated one-bond factor for an extensible freely jointed bond.

    The calculation is written in nondimensional form. The argument s may be
    real or complex. In the fixed-extension calculation, this function is
    evaluated at s = i q.

    Parameters
    ----------
    s : real or complex
        Dimensionless transform variable.
    kl : real
        Dimensionless bond-stretching stiffness.

    Returns
    -------
    Qbar : mp.mpf or mp.mpc
        Angle-integrated one-bond factor.
    """

    s  = mp.mpc(s) if isinstance(s, (complex, mp.mpc)) else mp.mpf(s)
    kl = mp.mpf(kl)
    alpha = mp.sqrt(kl / 2)

    # Use the continuous limit at s = 0 to avoid the apparent 1/s singularity.
    if abs(s) < mp.mpf("1e-30"):
        return (
            mp.sqrt(mp.pi) * (1 + 2 * alpha**2) / (2 * alpha**3)
        ) * mp.erfc(-alpha) + mp.exp(-alpha**2) / alpha**2

    l_plus = 1 + s / kl
    l_minus = 1 - s / kl

    bracket = (
        l_plus * mp.exp(s) * mp.erfc(-alpha * l_plus)
        - l_minus * mp.exp(-s) * mp.erfc(-alpha * l_minus)
    )

    prefactor = (
        mp.sqrt(mp.pi)
        / (2 * alpha * s)
        * mp.exp(s**2 / (2 * kl))
    )

    return prefactor * bracket


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


def FE_FJB_rigid(N, nz, dz, mp_dps):
    """
    Fixed-extension force-extension relation for rigid freely jointed bonds.

    For one rigid freely jointed bond, the Fourier-space one-bond factor is
        Qhat(q) = 2 sin(q) / q,

    with Qhat(0) = 2 by continuity. The N-bond Fourier-space partition
    function is obtained as Qhat(q)^N, followed by an inverse Fourier transform
    to fixed-extension space.

    Parameters
    ----------
    N : int
        Number of bonds.
    nz : int
        Number of Fourier grid points. Must be even.
    dz : float
        Extension-grid spacing.
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
        # For rigid bonds, the physical support is |R| <= N,
        # so L should be larger than N to avoid periodic aliasing.
        L = mp.mpf(nz) * dz / 2
        if L <= N:
            raise ValueError("The real-space half-domain L = nz*dz/2 must be larger than N.")

        dq = mp.pi / L
        k0 = nz // 2

        # Nonpositive q half-axis:
        # q = -qmax, ..., -dq, 0.
        q_nonpos = [
            dq * (mp.mpf(j) - k0)
            for j in range(k0 + 1)
        ]

        # One-bond Fourier factor for rigid FJB.
        Qhat_nonpos = [
            mp.mpf(2) if abs(q) < mp.mpf("1e-30") else 2 * mp.sin(q) / q
            for q in q_nonpos
        ]

        # Normalize before raising to the Nth power.
        # This only shifts H by an additive constant and does not affect fH.
        Qmax = max(abs(v) for v in Qhat_nonpos)
        Bhat_nonpos = [
            (v / Qmax) ** N
            for v in Qhat_nonpos
        ]

        # Inverse Fourier transform to fixed-extension space.
        z_pos, Bz_pos = inverse_fourier_qnonpos_to_zpos(
            Bhat_nonpos=Bhat_nonpos,
            dq=dq,
            dz=dz,
            nz=nz,
        )

        R, H, fH = Bz_to_FE_force(Bz_pos, z_pos)

    return R, H, fH


def FE_FJB_extensible(N, nz, dz, kl, mp_dps):
    """
    Fixed-extension force-extension relation for extensible freely jointed bonds.

    For one extensible freely jointed bond, the Fourier-space one-bond factor is
    Qbar(iq), where Qbar is the angle-integrated one-bond factor. The N-bond
    Fourier-space partition function is obtained as Qbar(iq)^N, followed by an
    inverse Fourier transform to fixed-extension space.

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
        kl = mp.mpf(kl)

        # The real-space computational interval is [-L, L).
        # For extensible bonds, the distribution is not compactly supported,
        # but L should still be chosen large enough that the tail is negligible
        # at the boundary.
        L = mp.mpf(nz) * dz / 2
        dq = mp.pi / L
        k0 = nz // 2

        # Nonpositive q half-axis:
        # q = -qmax, ..., -dq, 0.
        q_nonpos = [
            dq * (mp.mpf(j) - k0)
            for j in range(k0 + 1)
        ]

        # One-bond Fourier factor for extensible FJB.
        # In the fixed-extension calculation, s = i q.
        Qhat_nonpos = [
            Qbar_extensible_mp(mp.mpc(0, q), kl)
            for q in q_nonpos
        ]

        # Normalize before raising to the Nth power.
        # This only shifts H by an additive constant and does not affect fH.
        Qmax = max(abs(v) for v in Qhat_nonpos)
        Bhat_nonpos = [
            (v / Qmax) ** N
            for v in Qhat_nonpos
        ]

        # Inverse Fourier transform to fixed-extension space.
        z_pos, Bz_pos = inverse_fourier_qnonpos_to_zpos(
            Bhat_nonpos=Bhat_nonpos,
            dq=dq,
            dz=dz,
            nz=nz,
        )

        R, H, fH = Bz_to_FE_force(Bz_pos, z_pos)

    return R, H, fH



if __name__ == '__main__':
    
    kl = 500             # dimensionless bond-stretching stiffness
    
    # Small example for demonstration. Increase N, nz and mp_dps for production calculations.
    N = 4                # number of bonds
    nz = 1024            # number of Fourier grid points; must be even
    dz = 0.01            # extension-grid spacing

    mp_dps = 80          # decimal precision used by mpmath


    R_rigid, H_rigid, fH_rigid = FE_FJB_rigid(N, nz, dz, mp_dps)
    R_ext, H_ext, fH_ext = FE_FJB_extensible(N, nz, dz, kl, mp_dps)


    plt.figure()
    plt.rc('font',family='serif',size=14)
    plt.plot(R_rigid/N, fH_rigid, '-', linewidth=1, label='Rigid Bonds')
    plt.plot(R_ext/N, fH_ext, '-', linewidth=1, label='Extensible Bonds')
    plt.xlim([-0.05,1.25])
    plt.xticks(np.arange(0,1.25,step=0.2))
    plt.ylim([-5,95])
    plt.yticks(np.arange(0,100,step=30))
    plt.xlabel(r'$R/(Nl_0)$')
    plt.ylabel(r'$\beta fl_e$')
    plt.legend(loc='upper left')
    plt.grid(False)
    plt.show()