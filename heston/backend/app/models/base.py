"""
Abstract base class for volatility models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel
import numpy as np


@dataclass
class Greeks:
    """Option Greeks calculation results."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class CalibrationResult:
    """Result of model calibration to market data."""
    parameters: Dict[str, float]
    objective_value: float
    convergence: bool
    iterations: int
    fit_quality: float  # R-squared or similar metric
    errors: List[float]


class ModelParameters(BaseModel):
    """Base class for model parameters."""
    pass


class AbstractVolatilityModel(ABC):
    """
    Abstract base class for all volatility models.
    
    Provides standardized interface for option pricing, calibration,
    and Greek calculations across different volatility models.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = None
        
    @abstractmethod
    def price_option(self, spot: float, strike: float, expiry: float, 
                    risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """
        Price a European option using the model.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Option price
        """
        pass
    
    @abstractmethod
    def implied_volatility(self, spot: float, strike: float, expiry: float,
                          option_price: float, risk_free_rate: float,
                          dividend_yield: float = 0.0, option_type: str = 'call') -> float:
        """
        Calculate implied volatility from option price.
        
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
        pass
    
    @abstractmethod
    def calculate_greeks(self, spot: float, strike: float, expiry: float,
                        risk_free_rate: float, dividend_yield: float = 0.0) -> Greeks:
        """
        Calculate option Greeks.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            Greeks object with delta, gamma, theta, vega, rho
        """
        pass
    
    @abstractmethod
    def calibrate(self, market_data: Dict[str, Any]) -> CalibrationResult:
        """
        Calibrate model parameters to market data.
        
        Args:
            market_data: Dictionary containing market data with strikes, expiries, prices
            
        Returns:
            CalibrationResult with fitted parameters and quality metrics
        """
        pass
    
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get valid parameter bounds for the model.
        
        Returns:
            Dictionary mapping parameter names to (min, max) bounds
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """
        Validate that parameters are within reasonable bounds.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            True if parameters are valid, False otherwise
        """
        bounds = self.get_parameter_bounds()
        for param, value in parameters.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def generate_volatility_surface(self, spot: float, strikes: List[float], 
                                   expiries: List[float], risk_free_rate: float,
                                   dividend_yield: float = 0.0) -> np.ndarray:
        """
        Generate implied volatility surface for given strikes and expiries.
        
        Args:
            spot: Current spot price
            strikes: List of strike prices
            expiries: List of time to expirations
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield (default: 0.0)
            
        Returns:
            2D numpy array of implied volatilities [strikes x expiries]
        """
        surface = np.zeros((len(strikes), len(expiries)))
        
        for i, strike in enumerate(strikes):
            for j, expiry in enumerate(expiries):
                try:
                    # Price the option
                    option_price = self.price_option(spot, strike, expiry, 
                                                   risk_free_rate, dividend_yield)
                    
                    # Calculate implied volatility
                    iv = self.implied_volatility(spot, strike, expiry, option_price,
                                               risk_free_rate, dividend_yield)
                    surface[i, j] = iv
                except:
                    # Handle numerical issues
                    surface[i, j] = np.nan
                    
        return surface 