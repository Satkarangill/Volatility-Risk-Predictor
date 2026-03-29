"""
Risk Parity portfolio optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict, Tuple


def compute_risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute risk contributions of each asset.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns:
    --------
    np.ndarray : Risk contributions
    """
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    if portfolio_vol == 0:
        return np.zeros(len(weights))
    
    # Marginal contribution to risk
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    
    # Risk contribution = weight * marginal contribution
    risk_contrib = weights * marginal_contrib
    
    return risk_contrib


def risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Risk parity objective: minimize sum of squared differences in risk contributions.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns:
    --------
    float : Objective value
    """
    risk_contrib = compute_risk_contributions(weights, cov_matrix)
    
    # Target: equal risk contribution for each asset
    target_risk = risk_contrib.sum() / len(weights)
    
    # Sum of squared differences
    objective = np.sum((risk_contrib - target_risk) ** 2)
    
    return objective


def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    constraints: Optional[Dict] = None
) -> Tuple[pd.Series, Dict]:
    """
    Optimize risk parity portfolio.
    
    Parameters:
    -----------
    cov_matrix : pd.DataFrame
        Covariance matrix
    constraints : Dict, optional
        Additional constraints
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    n_assets = len(cov_matrix)
    
    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets
    
    # Constraints: weights sum to 1
    constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    if constraints:
        if 'long_short' in constraints and constraints['long_short']:
            bounds = tuple((-1, 1) for _ in range(n_assets))
        if 'max_weight' in constraints:
            bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
    
    # Objective function
    objective_func = lambda w: risk_parity_objective(w, cov_matrix.values)
    
    # Optimize
    result = minimize(
        objective_func,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. {result.message}")
    
    optimal_weights = pd.Series(result.x, index=cov_matrix.index)
    
    # Calculate portfolio metrics
    risk_contrib = compute_risk_contributions(optimal_weights.values, cov_matrix.values)
    portfolio_vol = np.sqrt(np.dot(optimal_weights.values, np.dot(cov_matrix.values, optimal_weights.values)))
    
    optimization_results = {
        'portfolio_volatility': portfolio_vol,
        'risk_contributions': pd.Series(risk_contrib, index=cov_matrix.index),
        'optimization_success': result.success,
        'optimization_message': result.message
    }
    
    return optimal_weights, optimization_results


def inverse_volatility_weights(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Simple inverse volatility weighting (naive risk parity).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    window : int
        Rolling window for volatility calculation
    
    Returns:
    --------
    pd.Series : Portfolio weights
    """
    vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    
    return weights

