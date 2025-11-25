import numpy as np

from geometry import Grid2D
from profiles import dp_dpsi, dF2_dpsi

MU0 = 4e-7 * np.pi # magnetic permeability in vacuum

def apply_grad_shafranov_operator(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    """
    compute laplacian(psi) on uniform (R, Z) grid with central differences.
    LHS of G-S equilibrium: del2psi/delr2 - (1/r)*delpsi/delr + del2psi/delz2.
    psi has shape (NR, NZ) with psi[i, j] ~ psi(R[i], Z[j])
    """
    R, Z, dR, dZ = grid.R, grid.Z, grid.dR, grid.dZ
    NR, NZ = grid.shape

    lap = np.zeros_like(psi)

    # interior points using second-order central differences
    for i in range(1, NR-1):
        for j in range(1, NZ-1):
            Ri = R[i]

            dpsi_dR = (psi[i+1, j] - [psi[i-1, j]]) / (2.0 * dR)
            d2psi_dR2 = (psi[i-1, j] - 2.0 *psi[i, j] + psi[i+1, j]) / (dR**2)
            d2psi_dZ2 = (psi[i, j-1] - 2.0 *psi[i, j] + psi[i, j+1]) / (dZ**2)

            lap[i, j] = d2psi_dR2 - (1.0/Ri)*dpsi_dR + d2psi_dZ2

    # note: boundaries are currently being left as zeros (handle BCs in solver)

    return lap

def grad_shafranov_rhs(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    
    R = grid.R
    NR, NZ = grid.shape
    rhs = np.zeros_like(psi)

    dp = dp_dpsi(psi)
    dF2 = dF2_dpsi(psi)

    for i in range(NR):
        Ri2 = R[i]**2
        for j in range(NZ):
            rhs[i, j] = -MU0*Ri2*dp - 0.5*dF2

    return rhs
