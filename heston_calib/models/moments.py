# Moment-matching initialization for Heston model
# Estimates parameters from market cumulants to provide good starting point
import numpy as np
from scipy.optimize import least_squares
from typing import Tuple

def heston_cumulants(v0: float, kappa: float, theta: float, 
                     sigma: float, rho: float, T: float) -> np.ndarray:
    # Compute Heston model cumulants of log(S_T)
    # Returns [mean, variance, skewness, kurtosis] scaled by T
    
    # Mean of log(S_T/S_0) under risk-neutral measure
    c1 = -0.5 * v0 * T + 0.5 * kappa * theta * T
    
    # Variance
    c2 = v0 * T + kappa * theta * T**2 / 2
    if kappa != 0:
        c2 = (v0 / kappa) * (1 - np.exp(-kappa * T)) + \
             0.5 * kappa * theta * T**2
    
    # Skewness (scaled)
    if kappa != 0:
        c3 = (rho * sigma * v0 / (kappa**2)) * \
             (1 - np.exp(-kappa * T))**2 + \
             0.5 * rho * sigma * kappa * theta * T**3 / 3
    else:
        c3 = rho * sigma * v0 * T**2 / 2
    
    # Kurtosis (scaled)
    if kappa != 0:
        c4 = (sigma**2 * v0 / (2 * kappa**3)) * \
             (1 - np.exp(-kappa * T))**3 + \
             0.25 * sigma**2 * kappa * theta * T**4 / 12
    else:
        c4 = sigma**2 * v0 * T**3 / 6
    
    return np.array([c1, c2, c3, c4])

def estimate_market_cumulants(surface, T_ref: float = None) -> np.ndarray:
    # Estimate market cumulants from shortest liquid maturity
    # Uses simplified smile-based estimation
    if T_ref is None:
        T_ref = surface.T_list[0]  # Shortest maturity
    
    # Rough atm vol estimate
    atm_vol = 0.15
    if surface.iv is not None:
        atm_vol = np.median(surface.iv)
    
    # Estimate cumulants (simplified)
    c1 = 0.0  # Risk-neutral drift
    c2 = atm_vol**2 * T_ref  # Variance
    c3 = -0.01 * T_ref  # Typical negative skew for equity
    c4 = 0.001 * T_ref  # Excess kurtosis
    
    return np.array([c1, c2, c3, c4])

def moment_matching_init(surface, bounds: dict) -> dict:
    # Initialize Heston parameters via moment matching
    # Provides robust starting point for full calibration
    T_ref = surface.T_list[0]  # Use shortest maturity
    
    # Estimate market cumulants
    market_cum = estimate_market_cumulants(surface, T_ref)
    
    # Initial guess (conservative)
    x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.7])  # [v0, kappa, theta, sigma, rho]
    
    # Bounds arrays
    lb = np.array([bounds['v0'][0], bounds['kappa'][0], bounds['theta'][0],
                   bounds['sigma'][0], bounds['rho'][0]])
    ub = np.array([bounds['v0'][1], bounds['kappa'][1], bounds['theta'][1],
                   bounds['sigma'][1], bounds['rho'][1]])
    
    def residual(x):
        v0, kappa, theta, sigma, rho = x
        model_cum = heston_cumulants(v0, kappa, theta, sigma, rho, T_ref)
        
        # Weighted differences (emphasize variance and skew)
        weights = np.array([0.1, 1.0, 0.5, 0.1])
        diff = weights * (model_cum - market_cum)
        
        # Add soft Feller penalty
        feller_violation = max(0, sigma**2 - 2 * kappa * theta)
        diff = np.append(diff, 10.0 * feller_violation)
        
        return diff
    
    # Bounded least squares
    try:
        result = least_squares(residual, x0, bounds=(lb, ub), 
                              ftol=1e-4, max_nfev=100)
        v0, kappa, theta, sigma, rho = result.x
    except:
        # Fallback to sensible defaults
        v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    
    return {
        'v0': v0,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'rho': rho,
        'lam': 0.0
    }
