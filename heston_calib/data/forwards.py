# Forward price computation with dividend handling
import numpy as np

def compute_forward(S, r, q, T):
    # Compute forward price: F(T) = S * exp((r - q) * T)
    return S * np.exp((r - q) * T)

def compute_forwards_by_T(S, r, q, T_list):
    # Compute forwards for all maturities
    return {T: compute_forward(S, r, q, T) for T in T_list}
