# Staged optimization: moment-matching → global scan → local → polish
import numpy as np
from scipy.optimize import minimize, differential_evolution
from ..models.moments import moment_matching_init
from ..models.heston import HestonModel
from ..models.black_scholes import BlackScholesModel
from ..models.sabr import SABRModel
from .objective import create_objective
import time
from typing import Dict, Callable

def latin_hypercube_sample(bounds: list, n_samples: int) -> np.ndarray:
    # Generate Latin hypercube samples within bounds
    n_params = len(bounds)
    samples = np.zeros((n_samples, n_params))
    
    for i in range(n_params):
        lb, ub = bounds[i]
        # Generate n_samples evenly spaced intervals
        intervals = np.linspace(lb, ub, n_samples + 1)
        # Random sample within each interval
        for j in range(n_samples):
            samples[j, i] = np.random.uniform(intervals[j], intervals[j+1])
    
    # Shuffle each parameter independently
    for i in range(n_params):
        np.random.shuffle(samples[:, i])
    
    return samples

def calibrate_heston(surface, max_time: float = 300) -> Dict:
    # Staged Heston calibration
    # Stage 0: Moment matching seed
    # Stage 1: Global scan (Latin hypercube)
    # Stage 2: Local refinement (L-bfgs-b)
    # Stage 3: Polish with reduced penalty
    start_time = time.time()
    
    # Get bounds
    bounds_dict = HestonModel.get_bounds()
    bounds = [(bounds_dict['v0'][0], bounds_dict['v0'][1]),
              (bounds_dict['kappa'][0], bounds_dict['kappa'][1]),
              (bounds_dict['theta'][0], bounds_dict['theta'][1]),
              (bounds_dict['sigma'][0], bounds_dict['sigma'][1]),
              (bounds_dict['rho'][0], bounds_dict['rho'][1])]
    
    # Stage 0: Moment matching initialization
    print("Stage 0: Moment matching initialization...")
    init_params = moment_matching_init(surface, bounds_dict)
    x0 = np.array([init_params['v0'], init_params['kappa'], init_params['theta'],
                   init_params['sigma'], init_params['rho']])
    
    obj_func = create_objective(surface, 'heston', feller_lambda=1e2)
    init_obj = obj_func(x0)
    print(f"  Initial objective: {init_obj:.6f}")
    
    # Stage 1: Global scan with Latin hypercube
    print("Stage 1: Global scan...")
    n_samples = 32
    samples = latin_hypercube_sample(bounds, n_samples)
    
    # Add moment-matched seed to samples
    samples[0] = x0
    
    # Evaluate all samples
    sample_objs = np.array([obj_func(s) for s in samples])
    
    # Keep top 3 seeds
    top_k = 3
    top_indices = np.argsort(sample_objs)[:top_k]
    seeds = samples[top_indices]
    print(f"  Top seed objective: {sample_objs[top_indices[0]]:.6f}")
    
    # Stage 2: Local optimization from best seeds
    print("Stage 2: Local optimization (L-bfgs-b)...")
    best_result = None
    best_obj = np.inf
    
    for i, seed in enumerate(seeds):
        try:
            result = minimize(obj_func, seed, method='L-BFGS-B', bounds=bounds,
                            options={'ftol': 1e-8, 'maxiter': 200})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
                print(f"  Seed {i}: objective = {result.fun:.6f}")
        except:
            continue
    
    if best_result is None:
        best_result = type('obj', (object,), {'x': x0, 'fun': init_obj})()
    
    # Stage 3: Polishing pass (reduce Feller penalty)
    print("Stage 3: Polish with reduced penalty...")
    obj_func_polish = create_objective(surface, 'heston', feller_lambda=1e1)
    
    try:
        polish_result = minimize(obj_func_polish, best_result.x, method='L-BFGS-B',
                                bounds=bounds, options={'ftol': 1e-9, 'maxiter': 60})
        if polish_result.fun < best_obj:
            best_result = polish_result
            print(f"  Polished objective: {polish_result.fun:.6f}")
    except:
        pass
    
    elapsed = time.time() - start_time
    
    # Extract parameters
    v0, kappa, theta, sigma, rho = best_result.x
    
    return {
        'v0': v0,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'rho': rho,
        'lam': 0.0,
        'objective': best_result.fun,
        'time': elapsed
    }

def calibrate_bs(surface) -> Dict:
    # Simple Black-Scholes calibration
    start_time = time.time()
    
    bounds = [(0.01, 2.0)]
    x0 = [0.15]  # Initial guess
    
    obj_func = create_objective(surface, 'bs')
    
    result = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds,
                     options={'ftol': 1e-9})
    
    elapsed = time.time() - start_time
    
    return {
        'sigma': result.x[0],
        'objective': result.fun,
        'time': elapsed
    }

def calibrate_sabr(surface) -> Dict:
    # Sabr calibration with global+local
    start_time = time.time()
    
    bounds_dict = SABRModel.get_bounds()
    bounds = [(bounds_dict['alpha'][0], bounds_dict['alpha'][1]),
              (bounds_dict['beta'][0], bounds_dict['beta'][1]),
              (bounds_dict['rho'][0], bounds_dict['rho'][1]),
              (bounds_dict['nu'][0], bounds_dict['nu'][1])]
    
    obj_func = create_objective(surface, 'sabr')
    
    # Global optimization
    result = differential_evolution(obj_func, bounds, maxiter=100, seed=42)
    
    # Local refinement
    result_local = minimize(obj_func, result.x, method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-9})
    
    elapsed = time.time() - start_time
    
    return {
        'alpha': result_local.x[0],
        'beta': result_local.x[1],
        'rho': result_local.x[2],
        'nu': result_local.x[3],
        'objective': result_local.fun,
        'time': elapsed
    }
