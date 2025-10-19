# Evaluation metrics: R², Rmse, Mae, Mape
import numpy as np

def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray, 
                     weights: np.ndarray = None) -> float:
    # Weighted R²
    # R² = 1 - Σw(y - ŷ)² / Σw(y - ȳ)²
    if weights is None:
        weights = np.ones_like(y_true)
    
    # Weighted mean
    y_mean = np.average(y_true, weights=weights)
    
    # Sum of squares
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_mean)**2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                weights: np.ndarray = None) -> float:
    # Root mean squared error
    if weights is None:
        weights = np.ones_like(y_true)
    
    weights_norm = weights / weights.sum()
    mse = np.sum(weights_norm * (y_true - y_pred)**2)
    return np.sqrt(mse)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray,
               weights: np.ndarray = None) -> float:
    # Mean absolute error
    if weights is None:
        weights = np.ones_like(y_true)
    
    weights_norm = weights / weights.sum()
    mae = np.sum(weights_norm * np.abs(y_true - y_pred))
    return mae

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray,
                weights: np.ndarray = None) -> float:
    # Mean absolute percentage error
    if weights is None:
        weights = np.ones_like(y_true)
    
    # Avoid division by zero
    mask = y_true > 0.01
    if not mask.any():
        return np.nan
    
    weights_norm = weights[mask] / weights[mask].sum()
    mape = np.sum(weights_norm * np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape * 100  # As percentage

def evaluate_model(surface, model, params: dict) -> dict:
    # Compute all metrics for a calibrated model
    from ..models.heston import HestonModel
    from ..models.black_scholes import BlackScholesModel
    from ..models.sabr import SABRModel
    
    # Create model instance
    if isinstance(model, str):
        if model == 'heston':
            model_obj = HestonModel(params['v0'], params['kappa'], params['theta'],
                                   params['sigma'], params['rho'], params.get('lam', 0.0))
        elif model == 'bs':
            model_obj = BlackScholesModel(params['sigma'])
        elif model == 'sabr':
            model_obj = SABRModel(params['alpha'], params['beta'], 
                                 params['rho'], params['nu'])
    else:
        model_obj = model
    
    # Compute model prices
    model_prices = []
    for T in surface.T_list:
        K_arr = surface.K_by_T[T]
        if hasattr(model_obj, 'price'):
            prices_T = model_obj.price(surface.S, K_arr, T, surface.r, surface.q)
        else:
            prices_T = model_obj.price_call(surface.S, K_arr, T, surface.r, surface.q)
        model_prices.extend(prices_T)
    
    model_prices = np.array(model_prices)
    market_prices = surface.prices
    weights = surface.weights
    
    # Compute metrics
    r2 = compute_r_squared(market_prices, model_prices, weights)
    rmse = compute_rmse(market_prices, model_prices, weights)
    mae = compute_mae(market_prices, model_prices, weights)
    mape = compute_mape(market_prices, model_prices, weights)
    
    return {
        'r_squared': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'model_prices': model_prices
    }
