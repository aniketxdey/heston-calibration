# Sabr model with correct forward usage (not spot)
# Uses Hagan et al. (2002) approximation
import numpy as np
from typing import Tuple, Dict

class SABRModel:
    # Sabr stochastic volatility model
    # dF = α F^β dW1
    # dα = ν α dW2
    # cor(dW1, dW2) = ρ dt
    
    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        # Initialize Sabr model parameters
        # alpha: initial volatility
        # beta: cev exponent (0 ≤ β ≤ 1)
        # rho: correlation
        # nu: vol-of-vol
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def implied_vol(self, F: float, K: float, T: float) -> float:
        # Hagan approximation for implied volatility
        # Uses forward F, not spot
        if T <= 0 or F <= 0 or K <= 0:
            return 0.0
        
        # Atm case
        if abs(F - K) < 1e-6:
            return self.alpha / (F ** (1 - self.beta))
        
        # Log-moneyness
        log_FK = np.log(F / K)
        
        # Hagan formula components
        FK_avg = (F * K) ** ((1 - self.beta) / 2)
        
        # Z parameter
        z = (self.nu / self.alpha) * FK_avg * log_FK
        
        # X(z) with safe division
        if abs(z) < 1e-5:
            x_z = 1.0
        else:
            sqrt_term = np.sqrt(1 - 2 * self.rho * z + z**2)
            x_z = np.log((sqrt_term + z - self.rho) / (1 - self.rho))
            x_z = z / x_z
        
        # Leading term
        numerator = self.alpha
        denominator = FK_avg * (1 + (1 - self.beta)**2 / 24 * log_FK**2 + 
                                (1 - self.beta)**4 / 1920 * log_FK**4)
        
        # Time-dependent correction
        correction = 1 + T * ((1 - self.beta)**2 / 24 * self.alpha**2 / FK_avg**2 +
                             0.25 * self.rho * self.beta * self.nu * self.alpha / FK_avg +
                             (2 - 3 * self.rho**2) / 24 * self.nu**2)
        
        iv = (numerator / denominator) * x_z * correction
        
        return max(iv, 1e-4)
    
    def price_call(self, S: float, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
        # Price calls using Sabr implied vol + Black formula
        # Correctly uses forward F = S*exp((r-q)*T)
        if isinstance(K, (int, float)):
            K = np.array([K])
        
        # Forward (critical: use forward not spot)
        F = S * np.exp((r - q) * T)
        
        # Get Sabr implied vols
        ivs = np.array([self.implied_vol(F, k, T) for k in K])
        
        # Black formula with forward
        from scipy.stats import norm
        call_prices = []
        for k, iv in zip(K, ivs):
            if T <= 0 or iv <= 0:
                call_prices.append(max(S - k, 0))
                continue
            
            d1 = (np.log(F / k) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            call = np.exp(-r * T) * (F * norm.cdf(d1) - k * norm.cdf(d2))
            call_prices.append(max(call, 0.0))
        
        return np.array(call_prices)
    
    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        # Expanded bounds per design spec
        return {
            'alpha': (0.001, 5.0),
            'beta': (0.001, 0.999),
            'rho': (-0.999, 0.999),
            'nu': (0.001, 5.0)
        }
