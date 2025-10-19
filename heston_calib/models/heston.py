# Heston stochastic volatility model with characteristic function pricing
# Uses Carr-Madan fft for efficient vectorized pricing across strikes
import numpy as np
from scipy import integrate
from typing import Dict, Tuple
import warnings

class HestonModel:
    # Heston model with characteristic function-based pricing
    # dS = rS dt + √V S dW1
    # dV = κ(θ - V) dt + σ√V dW2
    # cor(dW1, dW2) = ρ dt
    
    def __init__(self, v0: float, kappa: float, theta: float, 
                 sigma: float, rho: float, lam: float = 0.0):
        # Initialize Heston model parameters
        # v0: initial variance
        # kappa: mean reversion speed
        # theta: long-term variance
        # sigma: vol-of-vol
        # rho: correlation
        # lam: market price of volatility risk (optional)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lam = lam
    
    def char_func(self, u: np.ndarray, T: float, S: float, r: float, q: float) -> np.ndarray:
        # Heston characteristic function using stable branches
        # Returns: E[exp(i*u*log(S_T))] under risk-neutral measure
        
        # Adjust kappa and theta for market price of risk
        kappa_star = self.kappa + self.lam
        theta_star = self.kappa * self.theta / kappa_star if kappa_star != 0 else self.theta
        
        # Complex frequency
        i = 1j
        
        # Stable branch (Little trap formulation)
        d = np.sqrt((self.rho * self.sigma * i * u - kappa_star)**2 + 
                    self.sigma**2 * (i * u + u**2))
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            g = (kappa_star - self.rho * self.sigma * i * u - d) / \
                (kappa_star - self.rho * self.sigma * i * u + d)
        
        # Time-dependent terms
        exp_dT = np.exp(-d * T)
        
        # C and D functions
        D = (kappa_star - self.rho * self.sigma * i * u - d) / self.sigma**2 * \
            (1 - exp_dT) / (1 - g * exp_dT)
        
        C = (r - q) * i * u * T + \
            kappa_star * theta_star / self.sigma**2 * \
            (T * (kappa_star - self.rho * self.sigma * i * u - d) - 
             2 * np.log((1 - g * exp_dT) / (1 - g)))
        
        # Characteristic function
        cf = np.exp(C + D * self.v0 + i * u * np.log(S))
        
        return cf
    
    def price_call_fft(self, S: float, K_vec: np.ndarray, T: float, 
                       r: float, q: float, N: int = 4096, 
                       alpha: float = 1.25) -> np.ndarray:
        # Price European calls using Carr-Madan fft
        # Vectorized across strikes for efficiency
        # K_vec: array of strike prices
        # N: number of fft points
        # alpha: damping factor for stability
        
        # Fft parameters
        eta = 0.25  # Grid spacing in u
        lam_spacing = 2 * np.pi / (N * eta)
        b = N * lam_spacing / 2
        
        # U grid
        u = np.arange(N) * eta
        
        # Log-strike grid
        log_strikes = np.log(K_vec)
        ku = -b + lam_spacing * np.arange(N)
        
        # Modified characteristic function for call option
        i = 1j
        cf_vals = self.char_func(u - (alpha + 1) * i, T, S, r, q)
        
        # Numerator in Carr-Madan formula
        with np.errstate(divide='ignore', invalid='ignore'):
            psi = np.exp(-r * T) * cf_vals / (alpha**2 + alpha - u**2 + i * (2 * alpha + 1) * u)
        
        # Simpson's rule weights
        simpson_w = np.ones(N) * (3 + (-1)**(np.arange(N) + 1))
        simpson_w[0] = 1
        simpson_w[-1] = 1
        simpson_w = simpson_w * eta / 3
        
        # Fft integrand
        x = np.exp(i * b * u) * psi * simpson_w
        
        # Apply fft
        fft_vals = np.fft.fft(x)
        
        # Interpolate to desired strikes
        call_values = np.interp(log_strikes, ku, 
                               np.real(fft_vals * np.exp(-alpha * ku) / np.pi))
        
        # Ensure non-negative
        call_values = np.maximum(call_values, 0.0)
        
        return call_values
    
    def price_call_integration(self, S: float, K: float, T: float, 
                                r: float, q: float) -> float:
        # Price single call via direct integration (fallback/validation)
        # Uses Lewis (2001) formula
        def integrand(u, K, S, T, r, q):
            cf = self.char_func(u - 0.5j, T, S, r, q)
            return np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))
        
        F = S * np.exp((r - q) * T)
        integral, _ = integrate.quad(integrand, 0, 100, args=(K, S, T, r, q))
        call = np.exp(-r * T) * (F - np.sqrt(K * F) / np.pi * integral)
        
        return max(call, 0.0)
    
    def price(self, S: float, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
        # Price calls for array of strikes at maturity T
        # Uses fft for efficiency
        if isinstance(K, (int, float)):
            K = np.array([K])
        
        try:
            prices = self.price_call_fft(S, K, T, r, q)
        except:
            # Fallback to integration if fft fails
            prices = np.array([self.price_call_integration(S, k, T, r, q) for k in K])
        
        return prices
    
    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        # Expanded parameter bounds per design spec
        return {
            'v0': (1e-4, 0.8),
            'kappa': (0.05, 25.0),
            'theta': (1e-4, 0.8),
            'sigma': (0.05, 5.0),
            'rho': (-0.995, 0.995),
            'lam': (-0.5, 0.5)
        }
    
    @staticmethod
    def feller_penalty(kappa: float, theta: float, sigma: float) -> float:
        # Soft Feller condition penalty
        # 2κθ ≥ σ² required for non-negative variance
        violation = max(0, sigma**2 - 2 * kappa * theta)
        return violation**2
