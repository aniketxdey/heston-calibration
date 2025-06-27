"""
Heston Model Implementation

Based on Heston (1993) with moment matching approximation for computational efficiency.
Implements the stochastic volatility model with mean-reverting variance process.
"""

import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
from .base import AbstractVolatilityModel, Greeks, CalibrationResult


class HestonParameters:
    """Heston model parameters container."""
    
    def __init__(self, v0: float = 0.04, kappa: float = 2.0, theta: float = 0.04, 
                 sigma: float = 0.3, rho: float = -0.7):
        """
        Initialize Heston parameters.
        
        Args:
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of volatility
            rho: Correlation between asset and variance
        """
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'v0': self.v0,
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho
        }
        
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'HestonParameters':
        """Create from dictionary."""
        return cls(**params)


class HestonModel(AbstractVolatilityModel):
    """
    Heston Stochastic Volatility Model Implementation.
    
    Based on Heston (1993) with moment matching approximation for computational efficiency.
    The model assumes the underlying asset follows:
    
    dS_t = (r - q)S_t dt + sqrt(V_t)S_t dW_1^t
    dV_t = kappa(theta - V_t)dt + sigma*sqrt(V_t)dW_2^t
    
    where dW_1^t * dW_2^t = rho * dt
    """
    
    def __init__(self):
        super().__init__("Heston")
        self.parameters = HestonParameters()
        
    def set_parameters(self, params: HestonParameters):
        """Set model parameters."""
        self.parameters = params
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get valid parameter bounds for Heston model."""
        return {
            'v0': (0.001, 1.0),      # Initial variance
            'kappa': (0.1, 10.0),    # Mean reversion speed
            'theta': (0.001, 1.0),   # Long-term variance
            'sigma': (0.1, 2.0),     # Volatility of volatility
            'rho': (-0.99, 0.99)     # Correlation
        }
        
    def _feller_condition(self) -> bool:
        """Check Feller condition: 2*kappa*theta >= sigma^2."""
        return 2 * self.parameters.kappa * self.parameters.theta >= self.parameters.sigma**2
        
    def _moment_matching_approximation(self, spot: float, strike: float, expiry: float,
                                      risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Moment matching approximation for Heston option pricing.
        
        This is a computationally efficient approximation that matches the first
        two moments of the Heston model to a log-normal distribution.
        """
        # Model parameters
        v0 = self.parameters.v0
        kappa = self.parameters.kappa
        theta = self.parameters.theta
        sigma = self.parameters.sigma
        rho = self.parameters.rho
        
        # Time to expiration
        T = expiry
        
        # Risk-neutral drift
        mu = risk_free_rate - dividend_yield
        
        # Expected variance at time T
        E_VT = theta + (v0 - theta) * np.exp(-kappa * T)
        
        # Variance of variance at time T
        Var_VT = (sigma**2 * v0 / kappa) * (1 - np.exp(-kappa * T))**2 + \
                 (sigma**2 * theta / (2 * kappa)) * (1 - np.exp(-2 * kappa * T))
        
        # Expected log-return
        E_log_return = (mu - 0.5 * E_VT) * T
        
        # Variance of log-return
        Var_log_return = E_VT * T + Var_VT * T**2 / 2
        
        # Effective volatility for Black-Scholes approximation
        sigma_eff = np.sqrt(Var_log_return / T)
        
        # Black-Scholes pricing with effective volatility
        d1 = (np.log(spot / strike) + (mu + 0.5 * sigma_eff**2) * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        
        call_price = spot * np.exp(-dividend_yield * T) * norm.cdf(d1) - \
                    strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
                    
        return call_price
        
    def price_option(self, spot: float, strike: float, expiry: float,
                    risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Price a European call option using Heston model.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Option price
        """
        return self._moment_matching_approximation(spot, strike, expiry, risk_free_rate, dividend_yield)
        
    def implied_volatility(self, spot: float, strike: float, expiry: float,
                          option_price: float, risk_free_rate: float,
                          dividend_yield: float = 0.0, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            option_price: Market option price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            option_type: 'call' or 'put' (default: 'call')
            
        Returns:
            Implied volatility
        """
        def black_scholes_price(sigma):
            """Black-Scholes price for given volatility."""
            d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * expiry) / (sigma * np.sqrt(expiry))
            d2 = d1 - sigma * np.sqrt(expiry)
            
            if option_type.lower() == 'call':
                return spot * np.exp(-dividend_yield * expiry) * norm.cdf(d1) - \
                       strike * np.exp(-risk_free_rate * expiry) * norm.cdf(d2)
            else:  # put
                return strike * np.exp(-risk_free_rate * expiry) * norm.cdf(-d2) - \
                       spot * np.exp(-dividend_yield * expiry) * norm.cdf(-d1)
                       
        def black_scholes_vega(sigma):
            """Black-Scholes vega for given volatility."""
            d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * expiry) / (sigma * np.sqrt(expiry))
            return spot * np.exp(-dividend_yield * expiry) * np.sqrt(expiry) * norm.pdf(d1)
            
        # Newton-Raphson iteration
        sigma = 0.3  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            price_diff = black_scholes_price(sigma) - option_price
            if abs(price_diff) < tolerance:
                break
                
            vega = black_scholes_vega(sigma)
            if abs(vega) < 1e-10:
                break
                
            sigma = sigma - price_diff / vega
            sigma = max(0.001, sigma)  # Ensure positive volatility
            
        return sigma
        
    def calculate_greeks(self, spot: float, strike: float, expiry: float,
                        risk_free_rate: float, dividend_yield: float = 0.0) -> Greeks:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Greeks object
        """
        # Finite difference step sizes
        ds = spot * 0.001  # 0.1% of spot
        dr = 0.001  # 10 basis points
        dt = 0.001  # 1 day
        
        # Current price
        price_0 = self.price_option(spot, strike, expiry, risk_free_rate, dividend_yield)
        
        # Delta and Gamma
        price_up = self.price_option(spot + ds, strike, expiry, risk_free_rate, dividend_yield)
        price_down = self.price_option(spot - ds, strike, expiry, risk_free_rate, dividend_yield)
        
        delta = (price_up - price_down) / (2 * ds)
        gamma = (price_up - 2 * price_0 + price_down) / (ds**2)
        
        # Theta
        price_theta = self.price_option(spot, strike, expiry - dt, risk_free_rate, dividend_yield)
        theta = (price_theta - price_0) / dt
        
        # Vega (with respect to v0)
        original_v0 = self.parameters.v0
        self.parameters.v0 += 0.001
        price_vega = self.price_option(spot, strike, expiry, risk_free_rate, dividend_yield)
        self.parameters.v0 = original_v0
        vega = (price_vega - price_0) / 0.001
        
        # Rho
        price_rho = self.price_option(spot, strike, expiry, risk_free_rate + dr, dividend_yield)
        rho = (price_rho - price_0) / dr
        
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
        
    def calibrate(self, market_data: Dict[str, Any]) -> CalibrationResult:
        """
        Calibrate Heston parameters to market data.
        
        Args:
            market_data: Dictionary containing:
                - 'strikes': List of strike prices
                - 'expiries': List of time to expirations
                - 'prices': 2D array of option prices [strikes x expiries]
                - 'spot': Current spot price
                - 'risk_free_rate': Risk-free rate
                - 'dividend_yield': Dividend yield
                
        Returns:
            CalibrationResult with fitted parameters
        """
        strikes = np.array(market_data['strikes'])
        expiries = np.array(market_data['expiries'])
        market_prices = np.array(market_data['prices'])
        spot = market_data['spot']
        risk_free_rate = market_data['risk_free_rate']
        dividend_yield = market_data.get('dividend_yield', 0.0)
        
        # Flatten data for optimization
        strikes_flat = []
        expiries_flat = []
        prices_flat = []
        
        for i, strike in enumerate(strikes):
            for j, expiry in enumerate(expiries):
                if not np.isnan(market_prices[i, j]) and market_prices[i, j] > 0:
                    strikes_flat.append(strike)
                    expiries_flat.append(expiry)
                    prices_flat.append(market_prices[i, j])
                    
        strikes_flat = np.array(strikes_flat)
        expiries_flat = np.array(expiries_flat)
        prices_flat = np.array(prices_flat)
        
        def objective_function(params):
            """Objective function for optimization."""
            v0, kappa, theta, sigma, rho = params
            
            # Set parameters
            self.parameters = HestonParameters(v0, kappa, theta, sigma, rho)
            
            # Calculate model prices
            model_prices = []
            for strike, expiry in zip(strikes_flat, expiries_flat):
                try:
                    price = self.price_option(spot, strike, expiry, risk_free_rate, dividend_yield)
                    model_prices.append(price)
                except:
                    model_prices.append(1e6)  # Penalty for failed pricing
                    
            model_prices = np.array(model_prices)
            
            # Calculate squared errors
            errors = (model_prices - prices_flat) / prices_flat  # Relative errors
            return np.sum(errors**2)
            
        # Parameter bounds
        bounds = [
            (0.001, 1.0),    # v0
            (0.1, 10.0),     # kappa
            (0.001, 1.0),    # theta
            (0.1, 2.0),      # sigma
            (-0.99, 0.99)    # rho
        ]
        
        # Initial guess
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
        
        # Optimization
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Extract results
        v0, kappa, theta, sigma, rho = result.x
        self.parameters = HestonParameters(v0, kappa, theta, sigma, rho)
        
        # Calculate final errors
        final_prices = []
        for strike, expiry in zip(strikes_flat, expiries_flat):
            price = self.price_option(spot, strike, expiry, risk_free_rate, dividend_yield)
            final_prices.append(price)
            
        final_prices = np.array(final_prices)
        errors = (final_prices - prices_flat) / prices_flat
        
        # Calculate R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum(((prices_flat - np.mean(prices_flat)) / prices_flat)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return CalibrationResult(
            parameters=self.parameters.to_dict(),
            objective_value=result.fun,
            convergence=result.success,
            iterations=result.nit,
            fit_quality=r_squared,
            errors=errors.tolist()
        ) 