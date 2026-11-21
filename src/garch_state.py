"""GARCH(1,1) 1-day anchor with explicit, storable state (Week 12).

Same model and schedule as src.analysis_w9.garch_anchors (the research
definition used in Weeks 9-11): percent log returns, constant mean, parameters
re-estimated every REFIT returns on ALL returns up to that day (expanding
window, first fit at MIN_OBS), and between refits the fitted (mu, omega,
alpha, beta) filter the realised returns day by day:

    refit day : (mu, omega, alpha, beta, h_t) = fit on r[0..i]
    otherwise : h_t = previous h_next
    h_next    = omega + alpha (r_t - mu)^2 + beta h_t          (= E_t[h_{t+1}])
    sigma_1d  = sqrt(252 h_next) / 100

The difference to the research function is that every row also stores the
state needed to continue the recursion (garch_i = index of the return in the
ticker's return series, the four parameters and h_next). The incremental
rebuild therefore restarts from the last row whose return is unchanged and
refits only on scheduled days, instead of refitting the whole history.
`path(returns)` reproduces analysis_w9.garch_anchors bit-for-bit
(analysis/week12 audit).
"""

import numpy as np
import pandas as pd

from src.analysis_w9 import H, MIN_OBS, REFIT, fit_garch

STATE_COLS = ["garch_i", "garch_mu", "garch_omega", "garch_alpha", "garch_beta", "garch_h_next"]
OUT_COLS = ["sigma_garch_1d", "sigma_garch_1y"] + STATE_COLS


def _sigma_1y(omega, alpha, beta, h_next):
    persist = alpha + beta
    if persist < 1:
        v = omega / (1 - persist)
        avg = v + (h_next - v) * (1 - persist ** H) / (H * (1 - persist))
    else:
        avg = h_next
    return np.sqrt(avg * 252) / 100.0


def path(log_ret: pd.Series, start: int = 0, seed: dict | None = None) -> pd.DataFrame:
    """GARCH outputs + state for log_ret (NaNs dropped) from position `start` on.

    log_ret : the ticker's FULL log-return history up to the last date to compute
              (a refit needs every return up to its day)
    start   : first position (in the NaN-free series) to compute
    seed    : state of position start-1 (keys of STATE_COLS); None -> start must be 0
    """
    r = log_ret.dropna() * 100.0
    vals = r.to_numpy(float)
    rows = np.full((len(r), len(OUT_COLS)), np.nan)
    mu = omega = alpha = beta = h_next = None
    if seed is not None and pd.notna(seed.get("garch_h_next")):
        mu, omega, alpha, beta, h_next = (seed["garch_mu"], seed["garch_omega"], seed["garch_alpha"],
                                          seed["garch_beta"], seed["garch_h_next"])
    elif start != 0:
        start = 0                                   # no usable state: recompute from the beginning
    for i in range(start, len(r)):
        if i + 1 < MIN_OBS:
            continue
        if h_next is None or (i + 1 - MIN_OBS) % REFIT == 0:
            mu, omega, alpha, beta, h_t = fit_garch(vals[:i + 1])
        else:
            h_t = h_next
        h_next = omega + alpha * (vals[i] - mu) ** 2 + beta * h_t
        rows[i] = [np.sqrt(h_next * 252) / 100.0, _sigma_1y(omega, alpha, beta, h_next),
                   i, mu, omega, alpha, beta, h_next]
    out = pd.DataFrame(rows[start:], index=r.index[start:], columns=OUT_COLS)
    return out


def continue_path(new_lr: pd.Series, seed: dict) -> pd.DataFrame | None:
    """Continue the recursion from a stored state over consecutive NEW returns without
    the return history. Returns None when a refit falls inside (the caller then needs
    the full history and uses path()). Same arithmetic as path()'s non-refit branch."""
    if seed is None or pd.isna(seed.get("garch_h_next")) or pd.isna(seed.get("garch_i")):
        return None
    r = new_lr.dropna() * 100.0
    i0 = int(seed["garch_i"]) + 1
    if any((i + 1 - MIN_OBS) % REFIT == 0 for i in range(i0, i0 + len(r))):
        return None
    mu, omega, alpha, beta, h_next = (seed["garch_mu"], seed["garch_omega"], seed["garch_alpha"],
                                      seed["garch_beta"], seed["garch_h_next"])
    rows = np.empty((len(r), len(OUT_COLS)))
    for j, rt in enumerate(r.to_numpy(float)):
        h_t = h_next
        h_next = omega + alpha * (rt - mu) ** 2 + beta * h_t
        rows[j] = [np.sqrt(h_next * 252) / 100.0, _sigma_1y(omega, alpha, beta, h_next),
                   i0 + j, mu, omega, alpha, beta, h_next]
    return pd.DataFrame(rows, index=r.index, columns=OUT_COLS)
