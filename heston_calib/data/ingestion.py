import pandas as pd
from typing import Optional

def load_csv(path: str, data_date: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {'strike': 'K', 'Strike': 'K', 'bid': 'bid', 'ask': 'ask',
               'lastPrice': 'last', 'Type': 'type', 'Expiration': 'expiration',
               'volume': 'volume', 'openInterest': 'OI', 'impliedVolatility': 'iv'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'bid' in df.columns and 'ask' in df.columns:
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['mid'] = df['mid'].fillna(df['last'])
    else:
        df['mid'] = df['last']
    current_date = pd.to_datetime(data_date) if data_date else pd.to_datetime('2025-05-23')
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['T'] = (df['expiration'] - current_date).dt.days / 365.25
    df = df[df['T'] > 0].copy()
    if 'type' in df.columns:
        df['type'] = df['type'].str.lower()
    return df

def extract_spot_rate_div(df, symbol='SPX'):
    S = df['S'].iloc[0] if 'S' in df.columns else 5977.35
    r = 0.045
    q = 0.0 if 'SPX' in symbol.upper() else 0.015
    return S, r, q
