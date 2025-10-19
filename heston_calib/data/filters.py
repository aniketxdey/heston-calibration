# Quality filters for option data
import numpy as np
import pandas as pd

def apply_quality_filters(df, moneyness_range=(0.95, 1.05), min_volume=50, min_oi=100, 
                          max_spread_pct=0.05, iv_range=(0.05, 0.50)):
    # Apply quality filters: moneyness, liquidity, spread, iv bounds, no-arbitrage
    df = df.copy()
    
    # Filter by moneyness (atm focus: 0.95-1.05)
    df['moneyness'] = df['K'] / df['S']
    df = df[(df['moneyness'] >= moneyness_range[0]) & (df['moneyness'] <= moneyness_range[1])]
    
    # Liquidity filter (volume >= 50 or open interest >= 100)
    has_volume = df['volume'] >= min_volume if 'volume' in df.columns else True
    has_oi = df['OI'] >= min_oi if 'OI' in df.columns else True
    df = df[has_volume | has_oi]
    
    # Spread filter (bid-ask spread <= 5% of mid)
    if 'bid' in df.columns and 'ask' in df.columns:
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
        df = df[df['spread_pct'] <= max_spread_pct]
    
    # Price sanity check (focus on reasonably priced options)
    df = df[(df['mid'] > 1.0) & (df['mid'] < 500.0)]
    
    # Implied volatility bounds
    if 'iv' in df.columns:
        df = df[(df['iv'] >= iv_range[0]) & (df['iv'] <= iv_range[1])]
    
    # No-arbitrage checks for calls
    if 'type' in df.columns:
        calls = df['type'] == 'call'
        df.loc[calls, 'intrinsic'] = np.maximum(df.loc[calls, 'S'] - df.loc[calls, 'K'], 0)
        df = df[~(calls & (df['mid'] < df['intrinsic'] - 0.01))]
    return df

def select_grid(df, n_maturities=8, max_strikes_per_T=20):
    # Select balanced maturity grid and strikes per maturity
    # Default: 8 maturities with 20 atm-focused strikes each
    T_groups = df.groupby('T')
    T_candidates = [(T, len(group)) for T, group in T_groups if len(group) >= 10]
    T_candidates.sort()
    n_mats = min(len(T_candidates), n_maturities)
    
    # Select evenly spaced maturities
    if n_mats > 0:
        indices = np.linspace(0, len(T_candidates)-1, n_mats, dtype=int)
        selected_T = [T_candidates[i][0] for i in indices]
    else:
        return pd.DataFrame()
    
    # For each maturity, select strikes closest to atm
    result = []
    for T in selected_T:
        group = df[df['T'] == T].copy()
        group['distance_to_atm'] = np.abs(group['moneyness'] - 1.0)
        group = group.sort_values('distance_to_atm').head(max_strikes_per_T)
        result.append(group)
    return pd.concat(result, ignore_index=True)
