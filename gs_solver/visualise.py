import numpy as np
import matplotlib.pyplot as plt

from geometry import Grid2D

def plot_psi_contours(psi: np.ndarray, grid: Grid2D, title: str = "ψ contours"):
    R, Z = np.meshgrid(grid.R, grid.Z, indexing="ij")
    plt.figure()
    cs = plt.contour(R, Z, psi, levels=20)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.show()
