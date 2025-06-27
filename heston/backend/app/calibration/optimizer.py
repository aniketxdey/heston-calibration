"""
Robust Optimization Framework

Multi-algorithm optimization framework that combines global search with
local refinement for reliable parameter estimation with constraint handling.
"""

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(self, parameters: Dict[str, float], objective_value: float,
                 convergence: bool, iterations: int, algorithm: str):
        self.parameters = parameters
        self.objective_value = objective_value
        self.convergence = convergence
        self.iterations = iterations
        self.algorithm = algorithm


class RobustOptimizer:
    """
    Robust optimization framework for model calibration.
    
    Implements multiple optimization strategies:
    1. Differential Evolution for global search
    2. L-BFGS-B for local refinement
    3. Multi-start optimization for robustness
    """
    
    def __init__(self):
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.n_restarts = 5
        
    def optimize(self, objective_function: Callable, 
                parameter_bounds: List[Tuple[float, float]],
                initial_guess: Optional[List[float]] = None,
                method: str = 'hybrid') -> OptimizationResult:
        """
        Optimize parameters using specified method.
        
        Args:
            objective_function: Function to minimize
            parameter_bounds: List of (min, max) bounds for each parameter
            initial_guess: Initial parameter values
            method: Optimization method ('differential_evolution', 'lbfgs', 'hybrid')
            
        Returns:
            OptimizationResult with best parameters
        """
        if method == 'differential_evolution':
            return self._differential_evolution_optimize(
                objective_function, parameter_bounds)
        elif method == 'lbfgs':
            return self._lbfgs_optimize(
                objective_function, parameter_bounds, initial_guess)
        elif method == 'hybrid':
            return self._hybrid_optimize(
                objective_function, parameter_bounds, initial_guess)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
    def _differential_evolution_optimize(self, objective_function: Callable,
                                       parameter_bounds: List[Tuple[float, float]]) -> OptimizationResult:
        """
        Global optimization using Differential Evolution.
        
        Args:
            objective_function: Function to minimize
            parameter_bounds: Parameter bounds
            
        Returns:
            OptimizationResult
        """
        try:
            result = differential_evolution(
                objective_function,
                parameter_bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                seed=42  # For reproducibility
            )
            
            return OptimizationResult(
                parameters=self._array_to_dict(result.x, parameter_bounds),
                objective_value=result.fun,
                convergence=result.success,
                iterations=result.nit,
                algorithm='differential_evolution'
            )
            
        except Exception as e:
            logger.error(f"Differential evolution failed: {e}")
            return self._create_failed_result('differential_evolution')
            
    def _lbfgs_optimize(self, objective_function: Callable,
                       parameter_bounds: List[Tuple[float, float]],
                       initial_guess: Optional[List[float]] = None) -> OptimizationResult:
        """
        Local optimization using L-BFGS-B.
        
        Args:
            objective_function: Function to minimize
            parameter_bounds: Parameter bounds
            initial_guess: Initial parameter values
            
        Returns:
            OptimizationResult
        """
        if initial_guess is None:
            # Use center of bounds as initial guess
            initial_guess = [(bounds[0] + bounds[1]) / 2 for bounds in parameter_bounds]
            
        try:
            result = optimize.minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=parameter_bounds,
                options={
                    'maxiter': self.max_iterations,
                    'gtol': self.tolerance
                }
            )
            
            return OptimizationResult(
                parameters=self._array_to_dict(result.x, parameter_bounds),
                objective_value=result.fun,
                convergence=result.success,
                iterations=result.nit,
                algorithm='lbfgs'
            )
            
        except Exception as e:
            logger.error(f"L-BFGS-B optimization failed: {e}")
            return self._create_failed_result('lbfgs')
            
    def _hybrid_optimize(self, objective_function: Callable,
                        parameter_bounds: List[Tuple[float, float]],
                        initial_guess: Optional[List[float]] = None) -> OptimizationResult:
        """
        Hybrid optimization: global search followed by local refinement.
        
        Args:
            objective_function: Function to minimize
            parameter_bounds: Parameter bounds
            initial_guess: Initial parameter values
            
        Returns:
            OptimizationResult
        """
        # Step 1: Global search with Differential Evolution
        global_result = self._differential_evolution_optimize(
            objective_function, parameter_bounds)
        
        if not global_result.convergence:
            logger.warning("Global optimization failed, trying local optimization")
            return self._lbfgs_optimize(objective_function, parameter_bounds, initial_guess)
            
        # Step 2: Local refinement with L-BFGS-B
        local_result = self._lbfgs_optimize(
            objective_function, parameter_bounds, list(global_result.parameters.values()))
            
        # Return the better result
        if local_result.objective_value < global_result.objective_value:
            return local_result
        else:
            return global_result
            
    def multi_start_optimize(self, objective_function: Callable,
                           parameter_bounds: List[Tuple[float, float]],
                           n_starts: Optional[int] = None) -> OptimizationResult:
        """
        Multi-start optimization for robustness.
        
        Args:
            objective_function: Function to minimize
            parameter_bounds: Parameter bounds
            n_starts: Number of starting points
            
        Returns:
            Best OptimizationResult from all starts
        """
        if n_starts is None:
            n_starts = self.n_restarts
            
        results = []
        
        # Generate random starting points
        for i in range(n_starts):
            initial_guess = self._generate_random_start(parameter_bounds)
            result = self._lbfgs_optimize(objective_function, parameter_bounds, initial_guess)
            results.append(result)
            
        # Find best result
        best_result = min(results, key=lambda r: r.objective_value)
        
        logger.info(f"Multi-start optimization completed with {n_starts} starts. "
                   f"Best objective value: {best_result.objective_value}")
        
        return best_result
        
    def _generate_random_start(self, parameter_bounds: List[Tuple[float, float]]) -> List[float]:
        """
        Generate random starting point within bounds.
        
        Args:
            parameter_bounds: Parameter bounds
            
        Returns:
            Random starting point
        """
        return [np.random.uniform(bounds[0], bounds[1]) for bounds in parameter_bounds]
        
    def _array_to_dict(self, array: np.ndarray, 
                      parameter_bounds: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Convert parameter array to dictionary.
        
        Args:
            array: Parameter array
            parameter_bounds: Parameter bounds for naming
            
        Returns:
            Parameter dictionary
        """
        # Generate parameter names based on bounds
        param_names = [f'param_{i}' for i in range(len(array))]
        
        return dict(zip(param_names, array))
        
    def _create_failed_result(self, algorithm: str) -> OptimizationResult:
        """
        Create failed optimization result.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Failed OptimizationResult
        """
        return OptimizationResult(
            parameters={},
            objective_value=np.inf,
            convergence=False,
            iterations=0,
            algorithm=algorithm
        )
        
    def validate_parameters(self, parameters: Dict[str, float],
                          parameter_bounds: List[Tuple[float, float]]) -> bool:
        """
        Validate that parameters are within bounds.
        
        Args:
            parameters: Parameter dictionary
            parameter_bounds: Parameter bounds
            
        Returns:
            True if parameters are valid
        """
        param_values = list(parameters.values())
        
        if len(param_values) != len(parameter_bounds):
            return False
            
        for value, (min_val, max_val) in zip(param_values, parameter_bounds):
            if not (min_val <= value <= max_val):
                return False
                
        return True 