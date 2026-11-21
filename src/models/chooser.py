"""Simple chooser option pricing: Rubinstein (1991) closed form + two-stage
GBM Monte Carlo with both choice rules (rational value-based vs. the paper's
moneyness rule).

Contract: at choice date t1 the holder decides whether the option is a
European call or put, both with strike K and expiry T2 (t1 < T2).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.models.bsm import _d1_d2, bs_call, bs_put


# ------------------------------------------------------------- closed form

def chooser_closed_form(S0, K, r, q, sigma, t1, T2):
    """Decomposition: call(K, T2) + e^{-q(T2-t1)} * put(K*, t1),
    with K* = K * exp(-(r-q)(T2-t1))   [Rubinstein 1991]."""
    tau = T2 - t1
    K_star = K * np.exp(-(r - q) * tau)
    return bs_call(S0, K, r, q, sigma, T2) + np.exp(-q * tau) * bs_put(
        S0, K_star, r, q, sigma, t1
    )


def chooser_closed_form_direct(S0, K, r, q, sigma, t1, T2):
    """Independent direct-N() formula (Hull) used as an internal cross-check
    of the decomposition implementation."""
    d1, d2 = _d1_d2(S0, K, r, q, sigma, T2)
    y1 = (np.log(S0 / K) + (r - q) * T2 + 0.5 * sigma**2 * t1) / (
        sigma * np.sqrt(t1)
    )
    y2 = y1 - sigma * np.sqrt(t1)
    return (
        S0 * np.exp(-q * T2) * norm.cdf(d1)
        - K * np.exp(-r * T2) * norm.cdf(d2)
        + K * np.exp(-r * T2) * norm.cdf(-y2)
        - S0 * np.exp(-q * T2) * norm.cdf(-y1)
    )


# ------------------------------------------------------------- Monte Carlo

@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int
    rule: str


def chooser_mc(S0, K, r, q, sigma, t1, T2, n_paths=100_000, rule="optimal",
               antithetic=True, seed=42) -> MCResult:
    """Two-stage GBM MC under the risk-neutral measure.

    rule="optimal":   at t1 choose max(C, P) (rational holder).
    rule="moneyness": the paper's rule — call iff S_t1 > K.

    Variance reduction: (a) antithetic variates on the t1 draw;
    (b) conditional MC — the t1-value of the chosen leg is priced with the
    BSM formula instead of simulating on to T2, removing all second-stage
    noise (legitimate because the chosen leg's t1-value IS its BSM price
    under the same dynamics).
    """
    rng = np.random.default_rng(seed)
    n_draw = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(n_draw)
    if antithetic:
        z = np.concatenate([z, -z])

    S_t1 = S0 * np.exp((r - q - 0.5 * sigma**2) * t1 + sigma * np.sqrt(t1) * z)
    tau = T2 - t1
    call_t1 = bs_call(S_t1, K, r, q, sigma, tau)
    put_t1 = bs_put(S_t1, K, r, q, sigma, tau)

    if rule == "optimal":
        v_t1 = np.maximum(call_t1, put_t1)
    elif rule == "moneyness":
        v_t1 = np.where(S_t1 > K, call_t1, put_t1)
    else:
        raise ValueError(f"unknown rule: {rule}")

    disc = np.exp(-r * t1) * v_t1
    return MCResult(
        price=float(disc.mean()),
        std_error=float(disc.std(ddof=1) / np.sqrt(len(disc))),
        n_paths=len(disc),
        rule=rule,
    )


def chooser_mc_terminal(S0, K, r, q, sigma, t1, T2, n_paths=100_000,
                        rule="optimal", seed=42) -> MCResult:
    """Full two-stage simulation to T2 (payoff-based, no conditional trick).
    Used to reproduce the paper's Table-3-style path table and as an
    independent check on chooser_mc."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths)
    S_t1 = S0 * np.exp((r - q - 0.5 * sigma**2) * t1 + sigma * np.sqrt(t1) * z1)
    tau = T2 - t1
    S_T2 = S_t1 * np.exp((r - q - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * z2)

    if rule == "optimal":
        choose_call = bs_call(S_t1, K, r, q, sigma, tau) >= bs_put(
            S_t1, K, r, q, sigma, tau
        )
    else:
        choose_call = S_t1 > K

    payoff = np.where(
        choose_call, np.maximum(S_T2 - K, 0.0), np.maximum(K - S_T2, 0.0)
    )
    disc = np.exp(-r * T2) * payoff
    return MCResult(
        price=float(disc.mean()),
        std_error=float(disc.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
        rule=rule,
    )
