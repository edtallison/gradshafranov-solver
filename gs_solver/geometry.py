from dataclasses import dataclass

import numpy as np

@dataclass
class Grid2D:
    R: np.ndarray # radial coords (1D, length NR)
    Z: np.ndarray # Z coords (1D, length NZ)
    dR: float
    dZ: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.R.size, self.Z.size)
    
def make_uniform_grid(
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    NR: int,
    NZ: int,
) -> Grid2D:
    R = np.linspace(R_min, R_max, NR)
    Z = np.linspace(Z_min, Z_max, NZ)

    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    return Grid2D(R, Z, dR, dZ)
