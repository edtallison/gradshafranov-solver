import numpy as np

from geometry import Grid2D
from operators import apply_grad_shafranov_operator, grad_shafranov_rhs

def solve_poisson_gauss_seidel(
    grid: Grid2D,
    rhs: np.ndarray,
    psi_bc: np.ndarray,
    max_iters: int = 10_000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    solve lap(psi) = rhs on the grid with simple gauss-seidel iteration.
    psi_bc is an array with boundary values (Dirichlet)
    """
    psi = psi_bc.copy()
    NR, NZ = grid.shape
    dR, dZ = grid.dR, grid.dZ
    R = grid.R

    for it in range(max_iters):
        psi_old = psi.copy()

        for i in range(1, NR-1):
            Ri = R[i]
            # this is based on rearranging laplacian finite differences for psi_ij:
            for j in range(1, NZ-1):
                coeff_centre = -2.0 / (dR**2) - 2.0 / (dZ**2)
                coeff_R = 1.0 / (dR**2)
                coeff_Z = 1.0 / (dZ**2)

                rhs_ij = rhs[i, j]

                psi[i, j] = (
                    -rhs_ij
                    - coeff_R * (psi[i+1, j] + psi[i-1, j])
                    - coeff_Z * (psi[i, j+1] + psi[i, j-1])
                ) / coeff_centre

        diff = np.linalg.norm(psi - psi_old) / (np.linalg.norm(psi_old) + 1e-12)
        if diff < tol:
            print(f"Gauss-Seidel converged in {it} iterations, rel change = {diff:.2e}")
            break

    return psi

def solve_grad_shafranov_fixed_point(
    grid: Grid2D,
    psi_init: np.ndarray,
    max_outer_iters: int = 50,
    inner_max_iters: int = 3_000,
    inner_tol: float = 1e-6,
) -> np.ndarray:
    psi = psi_init.copy()

    for k in range(max_outer_iters):
        rhs = grad_shafranov_rhs(psi, grid)
        psi_new = solve_poisson_gauss_seidel(
            grid, rhs, psi_bc=psi,
            max_iters=inner_max_iters, tol=inner_tol
        )
        diff = np.linalg.norm(psi_new-psi) / np.linalg.norm((psi) + 1e-12)
        print(f"[Outer {k}] fixed point rel change={diff:.2e}")
        psi=psi_new

        if diff<1e-3:
            print("Fixed point converged.")
            break

    return psi
