import numpy as np

from geometry import make_uniform_grid
from solver import solve_poisson_gauss_seidel
from visualise import plot_psi_contours

def main():
    grid = make_uniform_grid(
        R_min = 1.0, R_max = 2.0,
        Z_min = -0.5, Z_max = 0.5,
        NR = 64, NZ = 64,
    )

    NR, NZ = grid.shape

    # manufactured RHS e.g. uniform source inside domain
    rhs = np.zeros((NR, NZ))
    rhs[NR//4 : 3*NR//4, NZ//4 : 3*NZ//4] = 1.0

    # dirichlet BC: psi=0 on boundary
    psi_bc = np.zeros_like(rhs)

    psi = solve_poisson_gauss_seidel(
        grid=grid,
        rhs=rhs,
        psi_bc=psi_bc,
        max_iters=5000,
        tol=1e-5
    )

    plot_psi_contours(psi, grid, title="Poisson solution (test)")


if __name__ == "__main__":
    main()