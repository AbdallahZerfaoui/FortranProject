from imports import *

class ProblemAssembler:
    def __init__(self, grid: 'GridParameters', problem: 'ProblemDefinition'):
        self._grid = grid
        self._problem = problem
    
    def fill_rhs_steady(self, F: 'VectorBase'):
        """
        Fill the right-hand side vector F based on the problem definition.
        """
        _problem_size = self._grid.n
        _Nx = self._grid.config["Nx"]
        global_1D_indexes = np.arange(1, _problem_size + 1, dtype=int) #TODO: check if it is 0 to size or 1 to size + 1
        local_i_indexes = global_1D_indexes % _Nx
        local_j_indexes = global_1D_indexes // _Nx
        
        # Populate the right-hand side vector F based on the problem definition
        source_values_array = self._problem.eval_f(local_j_indexes * self._grid.dx,
												  local_i_indexes * self._grid.dy)
        F.data[:] = source_values_array
        

    
    
    