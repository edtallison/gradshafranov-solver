import numpy as np

from geometry import make_uniform_grid
from solver import solve_grad_shafranov_fixed_point
from visualise import plot_psi_contours

def main():
    grid = make_uniform_grid(
        R_min=1.0, R_max=2.0,
        Z_min=-0.5, Z_max=0.5,
        NR=64, NZ=64,
    )

    NR, NZ = grid.shape

    # initial guess for psi: something simple e.g. zero
    psi_init = np.zeros((NR, NZ))

    psi_eq = solve_grad_shafranov_fixed_point(
        grid=grid,
        psi_init=psi_init,
        max_outer_iters=20,
        inner_max_iters=4000,
        inner_tol=1e-5,
    )

    plot_psi_contours(psi_eq, grid, title="Grad-Shafranov equilibrium (very MVP)")


if __name__ == "__main__":
    main()
