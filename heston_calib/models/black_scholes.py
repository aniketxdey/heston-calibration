# Black-Scholes model with dividend yield support
import numpy as np
from scipy.stats import norm
from typing import Tuple

class BlackScholesModel:
    # Black-Scholes with constant volatility and dividends
    # dS = (r - q)S dt + σS dW
    
    def __init__(self, sigma: float):
        self.sigma = sigma
    
    def d1d2(self, S: float, K: float, T: float, r: float, q: float) -> Tuple[float, float]:
        # Compute d1 and d2 for Black-Scholes formula
        if T <= 0:
            return 0.0, 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2
    
    def price_call(self, S: float, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
        # Price European calls using analytical formula
        if isinstance(K, (int, float)):
            K = np.array([K])
        
        if T <= 0:
            return np.maximum(S - K, 0)
        
        d1, d2 = self.d1d2(S, K, T, r, q)
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return np.maximum(call, 0.0)
    
    def vega(self, S: float, K: float, T: float, r: float, q: float) -> float:
        # Vega for weighting purposes
        if T <= 0:
            return 0.0
        d1, _ = self.d1d2(S, K, T, r, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def get_bounds() -> dict:
        return {'sigma': (0.01, 2.0)}
