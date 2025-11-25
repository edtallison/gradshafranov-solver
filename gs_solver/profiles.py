import numpy as np

def pressure(psi: np.ndarray) -> np.ndarray:
    """
    p(psi): simple linear profile of pressure as a function of psi.
    """
    psi_max = np.max(psi)
    p0 = 1.0
    return p0 * (1.0 - psi / (psi_max + 1e-12))


def dp_psi(psi: np.ndarray) -> np.ndarray:
    """
    dp/dpsi: derivative of pressure profile wrt psi.
    for the simple example above, dp/dpsi = -p0 / psi_max (inside plasma), 0 outside.
    """
    psi_max = np.max(psi)
    p0 = 1.0
    base = -p0 / (psi_max + 1e-12)

    mask = (psi <= psi_max)
    out = np.zeros_like(psi)
    out[mask] = base
    return out

def F(psi: np.ndarray) -> np.ndarray:
    """
    F(psi) = R * B_phi.
    for v1, take F as constant
    """
    F0 = 1.0
    return F0 * np.ones_like(psi)

def dF2_dspi(psi: np.ndarray) -> np.ndarray:
    """
    d(F^2)/dpsi
    """
    return np.zeros_like(psi)
