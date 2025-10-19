# Main calibration script
# Loads data, builds surface, calibrates models, evaluates performance
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

# Add package to path
sys.path.insert(0, '/Users/aniketdey/Documents/GitHub/heston-calibration')

from heston_calib.data import load_csv, build_surface
from heston_calib.optimizers import calibrate_heston, calibrate_bs, calibrate_sabr
from heston_calib.eval import evaluate_model

def main():
    print("=" * 70)
    print("Heston calibration")
    print("=" * 70)
    
    # Load real market data
    print("\n[1/5] Loading and filtering data...")
    df = load_csv('heston_calib/data/all_options_all_dates.csv', data_date='2025-05-23')
    
    # Focus on calls only
    df = df[df['type'] == 'call'].copy()
    print(f"  Loaded {len(df)} call options")
    
    # Build surface
    print("\n[2/5] Building calibration surface...")
    surface = build_surface(df, symbol='SPX')
    print(f"  Surface: {len(surface.T_list)} maturities, {len(surface.prices)} points")
    print(f"  S={surface.S:.2f}, r={surface.r:.4f}, q={surface.q:.4f}")
    print(f"  T range: {surface.T_list[0]:.3f} to {surface.T_list[-1]:.3f} years")
    
    # Calibrate Heston (primary model)
    print("\n[3/5] Calibrating Heston model...")
    print("-" * 70)
    heston_params = calibrate_heston(surface)
    print(f"\nHeston parameters:")
    print(f"  v0 (initial var):    {heston_params['v0']:.6f}")
    print(f"  kappa (mean rev):    {heston_params['kappa']:.6f}")
    print(f"  theta (long var):    {heston_params['theta']:.6f}")
    print(f"  sigma (vol-of-vol):  {heston_params['sigma']:.6f}")
    print(f"  rho (correlation):   {heston_params['rho']:.6f}")
    print(f"  Time: {heston_params['time']:.2f}s")
    
    # Check Feller condition
    feller_lhs = 2 * heston_params['kappa'] * heston_params['theta']
    feller_rhs = heston_params['sigma']**2
    feller_ok = feller_lhs >= feller_rhs
    print(f"\n  Feller check: 2κθ = {feller_lhs:.6f} {'≥' if feller_ok else '<'} σ² = {feller_rhs:.6f} {'✓' if feller_ok else '✗'}")
    
    # Evaluate Heston
    print("\n[4/5] Evaluating models...")
    print("-" * 70)
    
    heston_metrics = evaluate_model(surface, 'heston', heston_params)
    print(f"\nHeston performance:")
    print(f"  R² =        {heston_metrics['r_squared']:.6f}")
    print(f"  Rmse =      ${heston_metrics['rmse']:.4f}")
    print(f"  Mae =       ${heston_metrics['mae']:.4f}")
    print(f"  Mape =      {heston_metrics['mape']:.2f}%")
    
    # Calibrate Black-Scholes (baseline)
    print("\n[5/5] Calibrating baseline models...")
    print("-" * 70)
    
    bs_params = calibrate_bs(surface)
    bs_metrics = evaluate_model(surface, 'bs', bs_params)
    
    print(f"\nBlack-Scholes:")
    print(f"  sigma = {bs_params['sigma']:.6f}")
    print(f"  R² =    {bs_metrics['r_squared']:.6f}")
    print(f"  Rmse =  ${bs_metrics['rmse']:.4f}")
    print(f"  Time:   {bs_params['time']:.2f}s")
    
    # Calibrate Sabr
    sabr_params = calibrate_sabr(surface)
    sabr_metrics = evaluate_model(surface, 'sabr', sabr_params)
    
    print(f"\nSabr:")
    print(f"  alpha = {sabr_params['alpha']:.6f}")
    print(f"  beta =  {sabr_params['beta']:.6f}")
    print(f"  rho =   {sabr_params['rho']:.6f}")
    print(f"  nu =    {sabr_params['nu']:.6f}")
    print(f"  R² =    {sabr_metrics['r_squared']:.6f}")
    print(f"  Rmse =  ${sabr_metrics['rmse']:.4f}")
    print(f"  Time:   {sabr_params['time']:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("Final summary")
    print("=" * 70)
    
    results = [
        ('Heston', heston_metrics['r_squared'], heston_metrics['rmse'], 
         heston_metrics['mape'], heston_params['time']),
        ('Black-Scholes', bs_metrics['r_squared'], bs_metrics['rmse'],
         bs_metrics['mape'], bs_params['time']),
        ('Sabr', sabr_metrics['r_squared'], sabr_metrics['rmse'],
         sabr_metrics['mape'], sabr_params['time'])
    ]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Model':<15} {'R²':>10} {'Rmse':>10} {'Mape':>10} {'Time':>10}")
    print("-" * 70)
    for model, r2, rmse, mape, t in results:
        print(f"{model:<15} {r2:>10.6f} ${rmse:>9.4f} {mape:>9.2f}% {t:>9.2f}s")
    
    print("\n" + "=" * 70)
    print(f"✓ Calibration complete - Heston R² = {heston_metrics['r_squared']:.3f}")
    print("=" * 70)
    
    return {
        'heston': (heston_params, heston_metrics),
        'bs': (bs_params, bs_metrics),
        'sabr': (sabr_params, sabr_metrics)
    }

if __name__ == '__main__':
    results = main()
