from typing import List, Tuple
from config import ConfigLoader


class GridParameters:
    def __init__(self, config: dict):
        self.config = config  # contains Nx, Ny, Lx, Ly, D
        self.dx = config["Lx"] / (config["Nx"] + 1)  # grid spacing in x direction
        self.dy = config["Ly"] / (config["Ny"] + 1)  # grid spacing in y direction
        self.n = config["Nx"] * config["Ny"]

    def global_index_to_grid(self, k: int) -> tuple:
        """
        Convert a global index to a grid index, it starts from (0, 0) to (Nx-1, Ny-1)
        """
        _Nx = self.config["Nx"]
        i = k % _Nx + 1
        j = k // _Nx + 1
        return (i, j)

    def grid_index_to_global(self, i: int, j: int) -> int:
        """
        Convert a grid index to a global index
        """
        _Nx = self.config["Nx"]
        k = (i - 1) + (j - 1) * _Nx
        return k
