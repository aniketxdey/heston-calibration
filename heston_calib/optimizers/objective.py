# Objective functions with vega weighting and penalties
# Supports price-based, iv-based, and hybrid objectives
import numpy as np
from ..models.heston import HestonModel
from ..models.black_scholes import BlackScholesModel
from ..models.sabr import SABRModel

class ObjectiveFunction:
    # Weighted objective with penalties for Heston calibration
    
    def __init__(self, surface, mode='price', feller_lambda=1e2):
        # Initialize objective function
        # mode: 'price' or 'iv'
        # feller_lambda: penalty weight for Feller condition
        self.surface = surface
        self.mode = mode
        self.feller_lambda = feller_lambda
        self.n_calls = 0
    
    def heston_objective(self, params: np.ndarray) -> float:
        # Objective for Heston model calibration
        # params = [v0, kappa, theta, sigma, rho, lam]
        self.n_calls += 1
        
        # Unpack parameters
        v0, kappa, theta, sigma, rho = params[:5]
        lam = params[5] if len(params) > 5 else 0.0
        
        # Create model
        model = HestonModel(v0, kappa, theta, sigma, rho, lam)
        
        # Compute model prices for all strikes
        model_prices = []
        idx = 0
        for T in self.surface.T_list:
            K_arr = self.surface.K_by_T[T]
            try:
                prices_T = model.price(self.surface.S, K_arr, T, 
                                      self.surface.r, self.surface.q)
                model_prices.extend(prices_T)
            except:
                # If pricing fails, return large penalty
                return 1e10
            idx += len(K_arr)
        
        model_prices = np.array(model_prices)
        
        # Weighted squared error
        errors = self.surface.prices - model_prices
        weighted_sse = np.sum(self.surface.weights * errors**2)
        
        # Feller penalty
        feller_pen = self.feller_lambda * model.feller_penalty(kappa, theta, sigma)
        
        return weighted_sse + feller_pen
    
    def bs_objective(self, params: np.ndarray) -> float:
        # Objective for Black-Scholes
        sigma = params[0]
        
        model = BlackScholesModel(sigma)
        
        model_prices = []
        for T in self.surface.T_list:
            K_arr = self.surface.K_by_T[T]
            prices_T = model.price_call(self.surface.S, K_arr, T,
                                       self.surface.r, self.surface.q)
            model_prices.extend(prices_T)
        
        model_prices = np.array(model_prices)
        errors = self.surface.prices - model_prices
        weighted_sse = np.sum(self.surface.weights * errors**2)
        
        return weighted_sse
    
    def sabr_objective(self, params: np.ndarray) -> float:
        # Objective for Sabr
        alpha, beta, rho, nu = params
        
        model = SABRModel(alpha, beta, rho, nu)
        
        model_prices = []
        for T in self.surface.T_list:
            K_arr = self.surface.K_by_T[T]
            try:
                prices_T = model.price_call(self.surface.S, K_arr, T,
                                           self.surface.r, self.surface.q)
                model_prices.extend(prices_T)
            except:
                return 1e10
        
        model_prices = np.array(model_prices)
        errors = self.surface.prices - model_prices
        weighted_sse = np.sum(self.surface.weights * errors**2)
        
        return weighted_sse

def create_objective(surface, model_type='heston', **kwargs):
    # Factory for objective functions
    obj = ObjectiveFunction(surface, **kwargs)
    
    if model_type == 'heston':
        return obj.heston_objective
    elif model_type == 'bs':
        return obj.bs_objective
    elif model_type == 'sabr':
        return obj.sabr_objective
    else:
        raise ValueError(f"Unknown model type: {model_type}")
