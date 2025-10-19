# Base classes and shared data structures
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

@dataclass
class CalibrationResult:
    success: bool
    params: Dict[str, float]
    objective_value: float
    iterations: int
    execution_time: float
    r_squared: float
    rmse: float
    mae: float
    mape: float
    
@dataclass
class Surface:
    # Market surface with quotes and weights
    S: float  # Spot price
    r: float  # Risk-free rate
    q: float  # Dividend yield
    T_list: np.ndarray  # Maturities
    K_by_T: Dict[float, np.ndarray]  # Strikes per maturity
    prices: np.ndarray  # Market prices
    weights: np.ndarray  # Vega/liquidity weights
    iv: Optional[np.ndarray] = None  # Implied vols if available
    forward_by_T: Optional[Dict[float, float]] = None  # Forwards per maturity
