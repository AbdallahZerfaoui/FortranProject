from imports import *

class ProblemDefinition():
    """
    Problem definition class for the PDE problem.
    This class defines the source term and boundary conditions
    for the PDE problem based on the specified case.
	"""
    def __init__(self, grid: GridParameters, case: str='steady_polynomial'):
        """
        Case:
		- steady polynomial source term : steady_polynomial
		- steady trigonometric source term : steady_trigonometric
		- unsteady
        """
        self._grid = grid
        self._case = case
        
    def eval_f(self, x: float, y: float, t: float=0.0) -> float:
        match self._case:
            case 'steady_polynomial':
                return self._steady_polynomial_source(x, y)
            case 'steady_trigonometric':
                return self._steady_trigonometric_source(x, y)
            case 'unsteady':
                return self._unsteady_source(x, y, t)
            case _:
                raise ValueError(f"Unknown case: {self._case}")
    
    def eval_g(self, x: float, y: float) -> float:
        match self._case:
            case 'steady_polynomial':
                return 0
            case 'steady_trigonometric':
                return self._steady_trigonometric_bc(x, y)
            case 'unsteady':
                return 0
            case _:
                raise ValueError(f"Unknown case: {self._case}")

    def eval_h(self, x: float, y: float) -> float:
        match self._case:
            case 'steady_polynomial':
                return 0
            case 'steady_trigonometric':
                return self._steady_trigonometric_bc(x, y)
            case 'unsteady':
                return 1
            case _:
                raise ValueError(f"Unknown case: {self._case}")
    
    # Helper functions
    def _steady_polynomial_source(self, x: float, y: float) -> float:
        """Steady state polynomial source term"""
        result = 2 * ((y - y**2) + (x - x**2))
        return result
    
    def _steady_trigonometric_source(self, x: float, y: float) -> float:
        """Steady state trigonometric source term"""
        result = np.sin(x) * np.cos(y)
        return result
    
    def _steady_trigonometric_bc(self, x: float, y: float) -> float:
        """Steady state trigonometric boundrie condition"""
        result = np.sin(x) * np.cos(y)
        return result
    
    def _unsteady_source(self, x: float, y: float, t: float) -> float:
        """Unsteady source term"""
        _Lx = self._grid.Lx
        _Ly = self._grid.Ly
    
        result = np.exp(-(x-_Lx/2)**2) * \
                np.exp(-(y-_Ly/2)**2) * \
                np.cos(np.pi * t / 2)
        return result