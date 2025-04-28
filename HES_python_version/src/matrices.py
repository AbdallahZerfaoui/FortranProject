import abc
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse 
from grid import GridParameters
from vectors import VectorBase, SequentialVector

class MatrixBase(abc.ABC):
    pass

class MatrixBase(abc.ABC):
    """
	Abstract base class defining the interface for matrix operations.
	Derived classes must implement these methods using appropriate
	"""
    def __init__():
        pass
    
    @abc.abstractmethod
    def multiply(self, u: 'VectorBase') -> 'VectorBase':
        """Multiplies the matrix with a vector and returns the result."""
        pass

    @abc.abstractmethod
    def populate(self, grid: GridParameters):
        """Populates the matrix with data based on the grid parameters."""
        pass
    

class SequentialSparseMatrix(MatrixBase):
	"""
	A sequential sparse matrix implementation using NumPy.
	"""
	def __init__(self, grid: GridParameters):
		self.grid = grid
		self._n = grid.n
		self.matrix = csr_matrix((self._n, self._n), dtype=np.float64)
  
	def multiply(self, u: 'SequentialVector') -> 'SequentialVector':
		"""
		Multiplies the sparse matrix with a vector and returns the result.
		"""
		product = self.matrix * u.data
		# Convert the result to a SequentialVector
		result = SequentialVector(self._n)
		result.data = product
		return result
      
	def populate(self):
		"""
		Populates the matrix with data based on the grid parameters
		using scipy.sparse.diags for efficiency.
		Represents a 2D finite difference discretization of D * Laplacian.
		"""
		_D = self.grid.config["D"]
		_Nx = self.grid.config["Nx"]
		_Ny = self.grid.config["Ny"]
              
		if self._n != _Nx * _Ny:
				raise ValueError("Total points _n does not match Nx * Ny")

		_dx = self.grid.dx
		_dy = self.grid.dy

		# Calculate the constant values for diagonals
		diag_val = 2.0 * _D * (1.0 / (_dx**2) + 1.0 / (_dy**2))
		offdiag_x = -_D / (_dx**2)
		offdiag_y = -_D / (_dy**2)

		# Create the arrays for the diagonals
		# Main diagonal (offset 0)
		diag0 = np.full(self._n, diag_val)

		# Off-diagonal for x-neighbors (offset +1 and -1)
		# Need to zero out elements that wrap around rows
		diag1 = np.full(self._n - 1, offdiag_x)
		# Indices where i % _Nx == _Nx - 1 (end of a row)
		# These correspond to indices _Nx-1, 2*_Nx-1, ... in the diag1 array
		end_of_row_indices = np.arange(_Nx - 1, self._n - 1, _Nx)
              
		diag1[end_of_row_indices] = 0.0

		# Off-diagonal for y-neighbors (offset +_Nx and -_Nx)
		diagN = np.full(self._n - _Nx, offdiag_y)

		# Assemble the diagonals list and offsets list for sparse.diags
		diagonals = [diagN, diag1, diag0, diag1, diagN]
		offsets = [-_Nx, -1, 0, 1, _Nx]

		# Create the sparse matrix (CSR format is often good for calculations)
		self.matrix = scipy.sparse.diags(
			diagonals, offsets, shape=(self._n, self._n), format='csr'
		)

	# def populate(self):
	# 	"""
	# 	Populates the matrix with data based on the grid parameters.
	# 	For simplicity, we will just fill it with random values.
	# 	"""
	# 	_D = self.grid.config["D"]
	# 	_Nx = self.grid.config["Nx"]
	# 	_dx = self.grid.dx
	# 	_dy = self.grid.dy

	# 	for i in range(self._n):
	# 		self.matrix[i, i] = 2.0 * _D * (1.0 / (_dx**2) + 1.0 / (_dy**2)) # diagonal
   
	# 		if i % _Nx != _Nx - 1:
	# 			self.matrix[i, i + 1] = -_D / (_dx**2) 
	# 			self.matrix[i + 1, i] = -_D / (_dx**2)
    
	# 		if (i + _Nx < self._n):
	# 			self.matrix[i, i + _Nx] = -_D / (_dy**2)
	# 			self.matrix[i + _Nx, i] = -_D / (_dy**2)
    
			