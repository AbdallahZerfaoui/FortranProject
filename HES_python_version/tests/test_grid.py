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
    assert grid.global_index_to_grid(3) == (4, 1)
    assert grid.global_index_to_grid(7) == (4, 2)
    assert grid.global_index_to_grid(9) == (2, 3)
    assert grid.global_index_to_grid(12) == (1, 4)
    assert grid.global_index_to_grid(15) == (4, 4)

def test_grid_index_to_global():
    config = {
        "Nx": 4,
        "Ny": 4,
        "Lx": 1.0,
        "Ly": 1.0,
        "D": 1.0
    }
    grid = GridParameters(config)
    assert grid.grid_index_to_global(1, 1) == 0
    assert grid.grid_index_to_global(4, 1) == 3
    assert grid.grid_index_to_global(4, 2) == 7
    assert grid.grid_index_to_global(2, 3) == 9
    assert grid.grid_index_to_global(1, 4) == 12
    assert grid.grid_index_to_global(4, 4) == 15