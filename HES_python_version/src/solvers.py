import abc
import numpy as np
import time
from vectors import VectorBase
from matrices import MatrixBase

class SolverBase(abc.ABC):
    pass

class SolverBase(abc.ABC):
	"""
	Abstract base class defining the interface for solvers.
	Derived classes must implement these methods using appropriate
	solving techniques (direct, iterative, etc.).
	"""
	@abc.abstractmethod
	def solve(self, A: 'MatrixBase', U: 'VectorBase', F: 'VectorBase') -> int:
		"""Solves the linear system AU = F and returns the number of iterations."""
		pass

class SequentialSolver(SolverBase):
	"""
	A sequential solver implementation using NumPy.
	"""
	def __init__(self, tolerance: float = 1e-4):
		self._epsilon = tolerance

	# def mesure_solver_speed(self, A: 'MatrixBase', U: 'VectorBase', F: 'VectorBase') -> float:
	# 	"""
	# 	Measures the time taken to solve the linear system AU = F.
	# 	"""
	# 	start_time = time.time()
	# 	self.solve(A, U, F)
	# 	end_time = time.time()
	# 	total_time = (end_time - start_time)
	# 	return total_time

	def solve(self, A: 'MatrixBase', U: 'VectorBase', F: 'VectorBase') -> int:
		"""
		Solves the linear system AU = F using a Conjugate gradient method.
		"""
		count = 0
		# U = SequentialVector(_n)
		# U.data = np.zeros(_n)
		# initialize residual
		R = A.multiply(U) # R = AU
		R.axpy(-1.0, F)   # R = AU - F

		# initialize direction
		DIR = R
		DIR.scale(-1.0)  # DIR = -R

		# Conjugate gradient loop
		norms_ratio = R.norm() / F.norm()
  
		while norms_ratio > self._epsilon:
			print(f"Iteration {count}: Residual norm ratio = {norms_ratio}")
			# compute matrix-vector product
			V = A.multiply(DIR)
   
			# compute alpha
			alpha = R.dot(R) / DIR.dot(V)

			# update solution: U = U + alpha * DIR
			U.axpy(alpha, DIR) 

			# update residual
			R_old = R
			R.axpy(alpha, V)

			# compute beta
			beta = R.dot(R) / R_old.dot(R_old)

			# update direction
			DIR.axpy(-beta, R)

			count += 1

			norms_ratio = R.norm() / F.norm()

		return count