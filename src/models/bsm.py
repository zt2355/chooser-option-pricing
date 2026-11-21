"""Black-Scholes-Merton vanilla pricing with continuous dividend yield."""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, q, sigma, T):
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bs_call(S, K, r, q, sigma, T):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, q, sigma, T):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
