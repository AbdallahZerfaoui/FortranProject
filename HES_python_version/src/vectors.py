import abc
import numpy as np

# Forward declaration for type hinting within methods
class VectorBase(abc.ABC): 
    """
    We declare the class here, because the type hunting in the class
    """
    pass

class VectorBase(abc.ABC):
    """
    Abstract base class defining the interface for vector operations.
    Derived classes must implement these methods using appropriate
    underlying data structures (NumPy array, distributed arrays, etc.).
    """

    # @abc.abstractmethod
    # def get_global_size(self) -> int:
    #     """Returns the total number of elements in the vector across all processes."""
    #     pass

    # @abc.abstractmethod
    # def get_local_size(self) -> int:
    #     """Returns the number of elements stored locally on this process."""
    #     pass

    # @abc.abstractmethod
    # def get_local_data(self) -> np.ndarray:
    #     """Provides access to the local NumPy array storing the vector data."""
    #     # This is the bridge to use NumPy's optimized functions
    #     pass

    @abc.abstractmethod
    def dot(self, other: 'VectorBase') -> float:
        """Computes the global dot product (self . other). Requires communication in parallel."""
        pass

    @abc.abstractmethod
    def axpy(self, alpha: float, y: 'VectorBase'):
        """Performs the operation self = self + alpha * y. Operates on local data."""
        # Uses NumPy's element-wise operations on local data
        pass

    @abc.abstractmethod
    def scale(self, alpha: float):
        """Performs the operation self = alpha * self. Operates on local data."""
        # Uses NumPy's element-wise scaling on local data
        pass

    @abc.abstractmethod
    def norm(self) -> float:
        """Computes the global L2 norm (sqrt(self . self)). Requires communication in parallel."""
        pass
    
    # @abc.abstractmethod
    # def copy(self, other: 'VectorBase'):
    #     """Copies the contents of other to self. Operates on local data."""
    #     # Uses NumPy's assignment/copy on local data
    #     pass


class SequentialVector(VectorBase):
    """
    Concrete implementation of VectorBase for sequential execution.
    Uses a single NumPy array to store the entire vector.
    """
    def __init__(self, global_size: int):
        self.data = np.ones(global_size, dtype=np.float64)
        self.global_size = global_size
        
    def dot(self, other: 'SequentialVector') -> float:
        """
        Computes the dot product with another SequentialVector.
        """
        return np.dot(self.data, other.data)
    
    def axpy(self, alpha: float, y: 'SequentialVector'):
        """
        Performs the operation self = self + alpha * y.
        """
        self.data += alpha * y.data
        
    def scale(self, alpha: float):
        """
        Performs the operation self = alpha * self.
        """
        self.data *= alpha
        
    def norm(self) -> float:
        """
        Computes the L2 norm of the vector.
        """
        vector_norm = np.linalg.norm(self.data)
        return vector_norm
    