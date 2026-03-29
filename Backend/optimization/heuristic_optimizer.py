"""
Heuristic optimization methods: genetic algorithms, simulated annealing, Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Optional
from scipy.optimize import differential_evolution, basinhopping
import warnings
warnings.filterwarnings('ignore')


def genetic_algorithm_optimize(
    objective_func: Callable,
    n_assets: int,
    bounds: Tuple = (0, 1),
    constraints: Optional[Callable] = None,
    popsize: int = 15,
    maxiter: int = 1000
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize using genetic algorithm (differential evolution).
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to minimize
    n_assets : int
        Number of assets
    bounds : Tuple
        Bounds for each weight
    constraints : Callable, optional
        Constraint function (should return 0 when satisfied)
    popsize : int
        Population size
    maxiter : int
        Maximum iterations
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    # Define bounds for all assets
    bounds_list = [bounds for _ in range(n_assets)]
    
    # Add constraint: weights sum to 1
    if constraints is None:
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    else:
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': constraints}
        ]
    
    # Run differential evolution
    result = differential_evolution(
        objective_func,
        bounds_list,
        constraints=constraints_list,
        popsize=popsize,
        maxiter=maxiter,
        seed=42
    )
    
    optimal_weights = result.x
    
    optimization_results = {
        'objective_value': result.fun,
        'optimization_success': result.success,
        'optimization_message': result.message,
        'n_iterations': result.nit
    }
    
    return optimal_weights, optimization_results


def simulated_annealing_optimize(
    objective_func: Callable,
    n_assets: int,
    x0: Optional[np.ndarray] = None,
    bounds: Tuple = (0, 1),
    n_iter: int = 1000,
    T: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize using simulated annealing.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to minimize
    n_assets : int
        Number of assets
    x0 : np.ndarray, optional
        Initial guess
    bounds : Tuple
        Bounds for each weight
    n_iter : int
        Number of iterations
    T : float
        Initial temperature
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    if x0 is None:
        x0 = np.ones(n_assets) / n_assets
    
    # Define bounds for all assets
    bounds_list = [bounds for _ in range(n_assets)]
    
    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Run basin hopping (simulated annealing variant)
    result = basinhopping(
        objective_func,
        x0,
        niter=n_iter,
        T=T,
        minimizer_kwargs={
            'method': 'SLSQP',
            'bounds': bounds_list,
            'constraints': constraints
        },
        seed=42
    )
    
    optimal_weights = result.x
    
    optimization_results = {
        'objective_value': result.fun,
        'optimization_success': result.success,
        'n_iterations': result.nit
    }
    
    return optimal_weights, optimization_results


def bayesian_optimization_optimize(
    objective_func: Callable,
    n_assets: int,
    bounds: Tuple = (0, 1),
    n_calls: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize using Bayesian optimization (scikit-optimize).
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to minimize
    n_assets : int
        Number of assets
    bounds : Tuple
        Bounds for each weight
    n_calls : int
        Number of function calls
    random_state : int
        Random seed
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        raise ImportError("scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
    
    # Define search space
    dimensions = [Real(bounds[0], bounds[1]) for _ in range(n_assets)]
    
    # Wrapper to ensure weights sum to 1
    def constrained_objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        return objective_func(weights)
    
    # Run Bayesian optimization
    result = gp_minimize(
        constrained_objective,
        dimensions,
        n_calls=n_calls,
        random_state=random_state
    )
    
    optimal_weights = np.array(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()  # Normalize
    
    optimization_results = {
        'objective_value': result.fun,
        'n_iterations': len(result.func_vals)
    }
    
    return optimal_weights, optimization_results


def optimize_portfolio_heuristic(
    objective_func: Callable,
    n_assets: int,
    method: str = 'genetic',
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize portfolio using heuristic methods.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to minimize
    n_assets : int
        Number of assets
    method : str
        'genetic', 'simulated_annealing', or 'bayesian'
    **kwargs
        Additional arguments for specific methods
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    if method == 'genetic':
        return genetic_algorithm_optimize(
            objective_func,
            n_assets,
            bounds=kwargs.get('bounds', (0, 1)),
            popsize=kwargs.get('popsize', 15),
            maxiter=kwargs.get('maxiter', 1000)
        )
    elif method == 'simulated_annealing':
        return simulated_annealing_optimize(
            objective_func,
            n_assets,
            x0=kwargs.get('x0', None),
            bounds=kwargs.get('bounds', (0, 1)),
            n_iter=kwargs.get('n_iter', 1000),
            T=kwargs.get('T', 1.0)
        )
    elif method == 'bayesian':
        return bayesian_optimization_optimize(
            objective_func,
            n_assets,
            bounds=kwargs.get('bounds', (0, 1)),
            n_calls=kwargs.get('n_calls', 100),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError("method must be 'genetic', 'simulated_annealing', or 'bayesian'")

