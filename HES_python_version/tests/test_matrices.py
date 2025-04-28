import pytest
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse
from grid import GridParameters
from src.vectors import SequentialVector
from src.matrices import SequentialSparseMatrix


def test_matrix_multiply():
    # Create a grid with specific parameters
    config = {"D": 1.0, "Nx": 4, "Ny": 4, "Lx": 1.0, "Ly": 1.0}
    grid = GridParameters(config)
    # grid.n = grid.config["Nx"] * grid.config["Ny"]

    # Initialize the matrix
    A = SequentialSparseMatrix(grid)
    A.populate()

    # Create a vector to multiply with
    u = SequentialVector(grid.n)
    u.data = np.random.rand(grid.n)

    v_too_long = SequentialVector(grid.n + 1)
    v_too_short = SequentialVector(grid.n - 1)
    v_too_long.data = np.random.rand(grid.n + 1)
    v_too_short.data = np.random.rand(grid.n - 1)

    # Perform the multiplication
    result_vector = A.multiply(u)

    # Check the shape of the result
    assert result_vector.data.shape == (grid.n,)

    # multiply with a vector of incorrect size
    with pytest.raises(ValueError) as excinfo:
        A.multiply(v_too_long)
    assert str(excinfo.value) == "Vector size does not match matrix size."

    with pytest.raises(ValueError) as excinfo:
        A.multiply(v_too_short)
    assert str(excinfo.value) == "Vector size does not match matrix size."

