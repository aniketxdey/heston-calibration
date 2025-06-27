"""
Black-Scholes-Merton Model Implementation

Classical Black-Scholes model with dividend yield support and analytical Greeks.
"""

import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
from .base import AbstractVolatilityModel, Greeks, CalibrationResult


class BlackScholesModel(AbstractVolatilityModel):
    """
    Black-Scholes-Merton Option Pricing Model.
    
    The classical model assumes the underlying asset follows geometric Brownian motion:
    dS_t = (r - q)S_t dt + sigma*S_t dW_t
    
    where r is the risk-free rate, q is the dividend yield, and sigma is volatility.
    """
    
    def __init__(self, volatility: float = 0.2):
        super().__init__("Black-Scholes")
        self.volatility = volatility
        
    def set_volatility(self, volatility: float):
        """Set model volatility."""
        self.volatility = volatility
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get valid parameter bounds for Black-Scholes model."""
        return {
            'volatility': (0.001, 2.0)  # Volatility
        }
        
    def price_option(self, spot: float, strike: float, expiry: float,
                    risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Price a European call option using Black-Scholes formula.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Option price
        """
        # Black-Scholes parameters
        S = spot
        K = strike
        T = expiry
        r = risk_free_rate
        q = dividend_yield
        sigma = self.volatility
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call option price
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
        
    def price_put_option(self, spot: float, strike: float, expiry: float,
                        risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Price a European put option using Black-Scholes formula.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Put option price
        """
        # Black-Scholes parameters
        S = spot
        K = strike
        T = expiry
        r = risk_free_rate
        q = dividend_yield
        sigma = self.volatility
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Put option price
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return put_price
        
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
            S = spot
            K = strike
            T = expiry
            r = risk_free_rate
            q = dividend_yield
            
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                
        def black_scholes_vega(sigma):
            """Black-Scholes vega for given volatility."""
            S = spot
            K = strike
            T = expiry
            r = risk_free_rate
            q = dividend_yield
            
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
            
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
        Calculate option Greeks analytically.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Greeks object
        """
        # Black-Scholes parameters
        S = spot
        K = strike
        T = expiry
        r = risk_free_rate
        q = dividend_yield
        sigma = self.volatility
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        delta = np.exp(-q * T) * norm.cdf(d1)
        
        # Gamma
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2) + 
                q * S * np.exp(-q * T) * norm.cdf(d1))
        
        # Vega
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
        
        # Rho
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
        
    def calibrate(self, market_data: Dict[str, Any]) -> CalibrationResult:
        """
        Calibrate Black-Scholes volatility to market data.
        
        Args:
            market_data: Dictionary containing:
                - 'strikes': List of strike prices
                - 'expiries': List of time to expirations
                - 'prices': 2D array of option prices [strikes x expiries]
                - 'spot': Current spot price
                - 'risk_free_rate': Risk-free rate
                - 'dividend_yield': Dividend yield
                
        Returns:
            CalibrationResult with fitted volatility
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
        
        def objective_function(sigma):
            """Objective function for optimization."""
            self.volatility = sigma
            
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
        bounds = [(0.001, 2.0)]  # Volatility bounds
        
        # Initial guess
        x0 = [0.2]
        
        # Optimization
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Extract results
        self.volatility = result.x[0]
        
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
            parameters={'volatility': self.volatility},
            objective_value=result.fun,
            convergence=result.success,
            iterations=result.nit,
            fit_quality=r_squared,
            errors=errors.tolist()
        ) 