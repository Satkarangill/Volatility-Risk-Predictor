"""
Modern Portfolio Theory (MPT) utilities for mean-variance optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict


def compute_expected_returns(returns: pd.DataFrame, method: str = 'mean') -> pd.Series:
    """
    Compute expected returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    method : str
        'mean' or 'exponential'
    
    Returns:
    --------
    pd.Series : Expected returns
    """
    if method == 'mean':
        expected_returns = returns.mean() * 252  # Annualize
    elif method == 'exponential':
        # Exponential weighted mean
        expected_returns = returns.ewm(span=252).mean().iloc[-1] * 252
    else:
        raise ValueError("method must be 'mean' or 'exponential'")
    
    return expected_returns


def compute_covariance_matrix(returns: pd.DataFrame, method: str = 'sample') -> pd.DataFrame:
    """
    Compute covariance matrix.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    method : str
        'sample', 'exponential', or 'shrinkage'
    
    Returns:
    --------
    pd.DataFrame : Covariance matrix
    """
    if method == 'sample':
        cov = returns.cov() * 252  # Annualize
    elif method == 'exponential':
        cov = returns.ewm(span=252).cov().iloc[-len(returns.columns):] * 252
    elif method == 'shrinkage':
        # Ledoit-Wolf shrinkage estimator
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        cov_array = lw.fit(returns).covariance_ * 252
        cov = pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)
    else:
        raise ValueError("method must be 'sample', 'exponential', or 'shrinkage'")
    
    return cov


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio variance."""
    return np.dot(weights, np.dot(cov_matrix, weights))


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Calculate portfolio expected return."""
    return np.dot(weights, expected_returns)


def negative_sharpe(weights: np.ndarray, expected_returns: np.ndarray, 
                   cov_matrix: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Negative Sharpe ratio (for minimization)."""
    port_return = portfolio_return(weights, expected_returns)
    port_variance = portfolio_variance(weights, cov_matrix)
    port_std = np.sqrt(port_variance)
    
    if port_std == 0:
        return 1e10
    
    sharpe = (port_return - risk_free_rate) / port_std
    return -sharpe


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    objective: str = 'max_sharpe',
    constraints: Optional[Dict] = None,
    risk_free_rate: float = 0.0
) -> Tuple[pd.Series, Dict]:
    """
    Optimize portfolio weights.
    
    Parameters:
    -----------
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    objective : str
        'max_sharpe', 'min_vol', or 'max_return'
    constraints : Dict, optional
        Additional constraints
    risk_free_rate : float
        Risk-free rate
    
    Returns:
    --------
    Tuple : (optimal_weights, optimization_results)
    """
    n_assets = len(expected_returns)
    
    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets
    
    # Constraints: weights sum to 1
    constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Add custom constraints
    if constraints:
        if 'long_short' in constraints and constraints['long_short']:
            # Allow short positions
            bounds = tuple((-1, 1) for _ in range(n_assets))
        if 'max_weight' in constraints:
            bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
    
    # Objective function
    if objective == 'max_sharpe':
        objective_func = lambda w: negative_sharpe(
            w, expected_returns.values, cov_matrix.values, risk_free_rate
        )
    elif objective == 'min_vol':
        objective_func = lambda w: portfolio_variance(w, cov_matrix.values)
    elif objective == 'max_return':
        objective_func = lambda w: -portfolio_return(w, expected_returns.values)
    else:
        raise ValueError("objective must be 'max_sharpe', 'min_vol', or 'max_return'")
    
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
    
    optimal_weights = pd.Series(result.x, index=expected_returns.index)
    
    # Calculate portfolio metrics
    port_return = portfolio_return(optimal_weights.values, expected_returns.values)
    port_vol = np.sqrt(portfolio_variance(optimal_weights.values, cov_matrix.values))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    optimization_results = {
        'portfolio_return': port_return,
        'portfolio_volatility': port_vol,
        'sharpe_ratio': sharpe,
        'optimization_success': result.success,
        'optimization_message': result.message
    }
    
    return optimal_weights, optimization_results


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_portfolios: int = 100,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Generate efficient frontier.
    
    Parameters:
    -----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    num_portfolios : int
        Number of portfolios to generate
    risk_free_rate : float
        Risk-free rate
    
    Returns:
    --------
    pd.DataFrame : Efficient frontier (return, volatility, sharpe, weights)
    """
    min_vol_weights, _ = optimize_portfolio(
        expected_returns, cov_matrix, objective='min_vol'
    )
    max_return_weights, _ = optimize_portfolio(
        expected_returns, cov_matrix, objective='max_return'
    )
    
    min_vol = portfolio_variance(min_vol_weights.values, cov_matrix.values) ** 0.5
    max_ret = portfolio_return(max_return_weights.values, expected_returns.values)
    
    target_returns = np.linspace(min_vol * expected_returns.mean(), max_ret, num_portfolios)
    
    frontier_data = []
    
    for target_return in target_returns:
        # Constraint: achieve target return
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, expected_returns.values) - target_return}
        ]
        
        result = minimize(
            lambda w: portfolio_variance(w, cov_matrix.values),
            x0=np.ones(len(expected_returns)) / len(expected_returns),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(len(expected_returns))),
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            vol = np.sqrt(portfolio_variance(weights, cov_matrix.values))
            ret = portfolio_return(weights, expected_returns.values)
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            
            frontier_data.append({
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                **{f'weight_{asset}': w for asset, w in zip(expected_returns.index, weights)}
            })
    
    return pd.DataFrame(frontier_data)

