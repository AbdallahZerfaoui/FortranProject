from imports import *

class Grid:
    def __init__(self, config: dict):
        self.config = config # contains Nx, Ny, Lx, Ly, D
        self.dx = config['Lx'] / (config['Nx'] + 1) # grid spacing in x direction
        self.dy = config['Ly'] / (config['Ny'] + 1) # grid spacing in y direction
        self.n = config['Nx'] * config['Ny']
        
    def global_index_to_grid(self, k: int) -> tuple:
        """Convert a global index to a grid index"""
        _Nx = self.config['Nx']
        i = k % _Nx
        j = k // _Nx
        return (i, j)

    def grid_index_to_global(self, i: int, j: int) -> int:
        """Convert a grid index to a global index"""
        _Nx = self.config['Nx']
        k = i + j * _Nx
        return k

    