# Vega-based weighting with liquidity and spread adjustments
import numpy as np
import pandas as pd
from scipy.stats import norm

def bs_vega(S, K, T, r, sigma, q=0.0):
    # Black-Scholes vega for weight computation
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def compute_weights(df, S, r, q):
    # Compute vega * liquidity * spread weights
    # Formula: w = vega * sqrt(oi + 1) * spread_penalty
    # Normalized per maturity then across surface
    
    # Estimate vega using atm vol approximation
    atm_vol = 0.15  # Typical for Spx
    if 'iv' in df.columns and df['iv'].notna().any():
        atm_vol = df['iv'].median()
    
    vegas = np.array([bs_vega(S, K, T, r, atm_vol, q) for K, T in zip(df['K'], df['T'])])
    
    # Liquidity boost from open interest
    liquidity = np.sqrt(df['OI'].values + 1) if 'OI' in df.columns else np.ones(len(df))
    
    # Spread penalty (down-weight wide spreads)
    spread_penalty = 1.0 / (1.0 + df['spread_pct'].values * 10) if 'spread_pct' in df.columns else np.ones(len(df))
    
    weights = vegas * liquidity * spread_penalty
    
    # Normalize per maturity
    df_temp = df.copy()
    df_temp['weight'] = weights
    for T in df_temp['T'].unique():
        mask = df_temp['T'] == T
        T_weights = weights[mask]
        if T_weights.sum() > 0:
            weights[mask] = T_weights / T_weights.sum()
    
    # Overall normalization
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    return weights
