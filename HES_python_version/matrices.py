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

	def populate(self):
		"""
		Populates the matrix with data based on the grid parameters.
		For simplicity, we will just fill it with random values.
		"""
		for i in range(self.n):
			for j in range(self.n):
				if np.random.rand() < 0.1:  # Sparse condition
					self.matrix[i, j] = np.random.rand()