from imports import *

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
  
	def multiply(self, u: 'VectorBase') -> 'VectorBase':
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
		Populates the matrix with data based on the grid parameters.
		For simplicity, we will just fill it with random values.
		"""
		_D = self.grid.config["D"]
		_Nx = self.grid.config["Nx"]
		_dx = self.grid.dx
		_dy = self.grid.dy

		for i in range(self._n):
			self.matrix[i, i] = 2.0 * _D * (1.0 / (_dx**2) + 1.0 / (_dy**2)) # diagonal
   
			if i % _Nx != _Nx - 1:
				self.matrix[i, i + 1] = -_D / (_dx**2) 
				self.matrix[i + 1, i] = -_D / (_dx**2)
    
			if (i + _Nx < self._n):
				self.matrix[i, i + _Nx] = -_D / (_dy**2)
				self.matrix[i + _Nx, i] = -_D / (_dy**2)
    
			