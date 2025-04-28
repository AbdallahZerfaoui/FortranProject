import pytest
from src.grid import GridParameters

def test_global_index_to_grid():
    config = {
        "Nx": 4,
        "Ny": 4,
        "Lx": 1.0,
        "Ly": 1.0,
        "D": 1.0
    }
    grid = GridParameters(config)
    assert grid.global_index_to_grid(0) == (1, 1)
    assert grid.global_index_to_grid(7) == (4, 2)
    assert grid.global_index_to_grid(9) == (2, 3)
    assert grid.global_index_to_grid(15) == (4, 4)
    