"""
SABR Model Implementation

Based on Hagan et al. (2002) with asymptotic expansion for implied volatility.
The SABR model is widely used in interest rate and FX markets.
"""

import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
from .base import AbstractVolatilityModel, Greeks, CalibrationResult


class SABRParameters:
    """SABR model parameters container."""
    
    def __init__(self, alpha: float = 0.2, beta: float = 0.5, rho: float = -0.1, nu: float = 0.3):
        """
        Initialize SABR parameters.
        
        Args:
            alpha: Initial volatility
            beta: CEV parameter (0 < beta <= 1)
            rho: Correlation between asset and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'nu': self.nu
        }
        
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'SABRParameters':
        """Create from dictionary."""
        return cls(**params)


class SABRModel(AbstractVolatilityModel):
    """
    SABR (Stochastic Alpha Beta Rho) Model Implementation.
    
    Based on Hagan et al. (2002). The model assumes:
    dF_t = alpha_t * F_t^beta * dW_1^t
    dalpha_t = nu * alpha_t * dW_2^t
    
    where dW_1^t * dW_2^t = rho * dt
    """
    
    def __init__(self):
        super().__init__("SABR")
        self.parameters = SABRParameters()
        
    def set_parameters(self, params: SABRParameters):
        """Set model parameters."""
        self.parameters = params
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get valid parameter bounds for SABR model."""
        return {
            'alpha': (0.001, 1.0),   # Initial volatility
            'beta': (0.01, 1.0),     # CEV parameter
            'rho': (-0.99, 0.99),    # Correlation
            'nu': (0.001, 2.0)       # Volatility of volatility
        }
        
    def _sabr_implied_volatility(self, spot: float, strike: float, expiry: float) -> float:
        """
        Calculate SABR implied volatility using Hagan's asymptotic expansion.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            
        Returns:
            Implied volatility
        """
        alpha = self.parameters.alpha
        beta = self.parameters.beta
        rho = self.parameters.rho
        nu = self.parameters.nu
        T = expiry
        
        # Forward price (assuming no dividends for simplicity)
        F = spot
        K = strike
        
        # Log-moneyness
        x = np.log(F / K)
        
        # z and chi
        z = (nu / alpha) * (F**(1 - beta) - K**(1 - beta)) / (1 - beta)
        chi = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        
        # Hagan's asymptotic expansion
        if abs(x) < 1e-8:  # At-the-money
            sigma_atm = alpha / (F**(1 - beta)) * (1 + 
                ((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2 * beta))) * T +
                (rho * beta * nu * alpha) / (4 * F**(1 - beta)) * T +
                ((2 - 3 * rho**2) / 24) * nu**2 * T)
        else:
            # Away from the money
            sigma = alpha / (F**(1 - beta)) * (x / chi) * (1 + 
                ((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2 * beta))) * T +
                (rho * beta * nu * alpha) / (4 * F**(1 - beta)) * T +
                ((2 - 3 * rho**2) / 24) * nu**2 * T)
            return sigma
            
        return sigma_atm
        
    def price_option(self, spot: float, strike: float, expiry: float,
                    risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Price a European call option using SABR model.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Option price
        """
        # Calculate SABR implied volatility
        sigma_iv = self._sabr_implied_volatility(spot, strike, expiry)
        
        # Use Black-Scholes with SABR implied volatility
        S = spot
        K = strike
        T = expiry
        r = risk_free_rate
        q = dividend_yield
        
        # Black-Scholes with SABR volatility
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma_iv**2) * T) / (sigma_iv * np.sqrt(T))
        d2 = d1 - sigma_iv * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
        
    def implied_volatility(self, spot: float, strike: float, expiry: float,
                          option_price: float, risk_free_rate: float,
                          dividend_yield: float = 0.0, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using SABR model.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            option_price: Market option price (not used in SABR)
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            option_type: 'call' or 'put' (default: 'call')
            
        Returns:
            SABR implied volatility
        """
        return self._sabr_implied_volatility(spot, strike, expiry)
        
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
        
        # Vega (with respect to alpha)
        original_alpha = self.parameters.alpha
        self.parameters.alpha += 0.001
        price_vega = self.price_option(spot, strike, expiry, risk_free_rate, dividend_yield)
        self.parameters.alpha = original_alpha
        vega = (price_vega - price_0) / 0.001
        
        # Rho
        price_rho = self.price_option(spot, strike, expiry, risk_free_rate + dr, dividend_yield)
        rho = (price_rho - price_0) / dr
        
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
        
    def calibrate(self, market_data: Dict[str, Any]) -> CalibrationResult:
        """
        Calibrate SABR parameters to market data.
        
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
            alpha, beta, rho, nu = params
            
            # Set parameters
            self.parameters = SABRParameters(alpha, beta, rho, nu)
            
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
            (0.001, 1.0),    # alpha
            (0.01, 1.0),     # beta
            (-0.99, 0.99),   # rho
            (0.001, 2.0)     # nu
        ]
        
        # Initial guess
        x0 = [0.2, 0.5, -0.1, 0.3]
        
        # Optimization
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Extract results
        alpha, beta, rho, nu = result.x
        self.parameters = SABRParameters(alpha, beta, rho, nu)
        
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