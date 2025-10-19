# Data handling: ingestion, filtering, forwards, weighting
from .ingestion import load_csv, extract_spot_rate_div
from .filters import apply_quality_filters, select_grid
from .forwards import compute_forward, compute_forwards_by_T
from .weights import compute_weights
from ..models.base import Surface
import numpy as np

def build_surface(df, S=None, r=None, q=None, symbol='SPX'):
    # Complete pipeline: load → filter → compute forwards → weight
    # Returns Surface object ready for calibration
    
    # Extract market parameters if not provided
    if S is None or r is None or q is None:
        S_est, r_est, q_est = extract_spot_rate_div(df, symbol)
        S = S or S_est
        r = r or r_est
        q = q or q_est
    
    # Add spot to dataframe for filters
    df['S'] = S
    
    # Apply quality filters
    df_filtered = apply_quality_filters(df)
    
    # Select balanced grid
    df_grid = select_grid(df_filtered)
    
    if len(df_grid) == 0:
        raise ValueError("No options survived filtering")
    
    # Compute forwards for all maturities
    T_list = np.sort(df_grid['T'].unique())
    forward_by_T = compute_forwards_by_T(S, r, q, T_list)
    
    # Group strikes by maturity
    K_by_T = {}
    for T in T_list:
        K_by_T[T] = np.sort(df_grid[df_grid['T'] == T]['K'].values)
    
    # Compute vega-based weights
    weights = compute_weights(df_grid, S, r, q)
    
    # Extract prices and implied vols
    prices = df_grid['mid'].values
    iv = df_grid['iv'].values if 'iv' in df_grid.columns else None
    
    return Surface(S=S, r=r, q=q, T_list=T_list, K_by_T=K_by_T,
                   prices=prices, weights=weights, iv=iv, forward_by_T=forward_by_T)
