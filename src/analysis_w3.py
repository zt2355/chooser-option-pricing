"""Week 3 analysis: BSM chooser replication package.

Produces (docs/week3/):
  figures/convergence.png        MC -> closed-form convergence (log-log)
  figures/sensitivity_panels.png sigma/K/r/q sensitivity (paper Fig.3-6 shapes)
  figures/t1_sensitivity.png     choice-date sensitivity (paper's open question)
  figures/rule_gap_vs_r.png      cost of the paper's moneyness rule vs r
  results_summary.json           headline numbers used in the weekly report

Usage: python -m src.analysis_w3
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.models.bsm import bs_call, bs_put
from src.models.chooser import (
    chooser_closed_form,
    chooser_closed_form_direct,
    chooser_mc,
    chooser_mc_terminal,
)

OUT = config.PROJECT_ROOT / "docs" / "week3"
FIG = OUT / "figures"

BLUE, ORANGE, GRAY, GREEN = "#2f6db3", "#d1603d", "#8a8a8a", "#3a7d44"
INK, GRID = "#333333", "#d9d9d9"


def load_params() -> dict:
    p = json.load(open(config.PROJECT_ROOT / "configs" / "paper_params.json"))
    p.pop("description", None)
    return p


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- pieces

def convergence_figure(p, cf):
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    cond, term = [], []
    for n in ns:
        cond.append(chooser_mc(**p, n_paths=n, seed=7))
        term.append(chooser_mc_terminal(**p, n_paths=n, seed=7))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ax.axhline(cf, color=GRAY, linewidth=1.2, linestyle="--")
    ax.annotate(f"closed form = {cf:.4f}", xy=(ns[0], cf), fontsize=8,
                color=GRAY, xytext=(0, 6), textcoords="offset points")
    for res, color, label in ((term, ORANGE, "terminal-payoff MC"),
                              (cond, BLUE, "conditional MC (variance-reduced)")):
        prices = [r.price for r in res]
        halfw = [2 * r.std_error for r in res]
        ax.errorbar(ns, prices, yerr=halfw, fmt="o-", color=color, label=label,
                    markersize=4, linewidth=1.6, capsize=3)
    ax.set_xscale("log")
    style(ax, "Monte Carlo convergence to the closed form (error bars = ±2 s.e.)",
          "number of paths (log scale)", "chooser price")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_paths": ns,
            "conditional": [[r.price, r.std_error] for r in cond],
            "terminal": [[r.price, r.std_error] for r in term]}


def sensitivity_figure(p):
    base = dict(p)
    grids = {
        "sigma": np.linspace(0.02, 1.0, 60),
        "K": np.linspace(60, 400, 80),
        "r": np.linspace(0.0, 0.10, 60),
        "q": np.linspace(0.0, 0.10, 60),
    }
    labels = {"sigma": r"volatility $\sigma$", "K": "strike $K$",
              "r": "risk-free rate $r$", "q": "dividend yield $q$"}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), dpi=200)
    for ax, (name, grid) in zip(axes.ravel(), grids.items()):
        ch, ca, pu = [], [], []
        for v in grid:
            pp = dict(base); pp[name] = v
            ch.append(chooser_closed_form(**pp))
            ca.append(bs_call(pp["S0"], pp["K"], pp["r"], pp["q"], pp["sigma"], pp["T2"]))
            pu.append(bs_put(pp["S0"], pp["K"], pp["r"], pp["q"], pp["sigma"], pp["T2"]))
        ax.plot(grid, ch, color=BLUE, linewidth=2, label="chooser")
        ax.plot(grid, ca, color=ORANGE, linewidth=1.4, label="call leg")
        ax.plot(grid, pu, color=GREEN, linewidth=1.4, label="put leg")
        ax.axvline(base[name], color=GRAY, linewidth=0.9, linestyle=":")
        style(ax, f"sensitivity to {labels[name]}", labels[name], "option value")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sensitivity replication (paper Fig.3-6 directions) — dotted line = paper's parameter",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_panels.png", bbox_inches="tight")
    plt.close(fig)


def t1_figure(p, cf):
    base = dict(p)
    t1s = np.linspace(0.01, base["T2"] - 0.005, 80)
    vals = [chooser_closed_form(**{**base, "t1": t}) for t in t1s]
    call = bs_call(base["S0"], base["K"], base["r"], base["q"], base["sigma"], base["T2"])
    put = bs_put(base["S0"], base["K"], base["r"], base["q"], base["sigma"], base["T2"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ax.plot(t1s, vals, color=BLUE, linewidth=2, label="chooser value")
    ax.axhline(max(call, put), color=ORANGE, linewidth=1.2, linestyle="--")
    ax.axhline(call + put, color=GREEN, linewidth=1.2, linestyle="--")
    ax.annotate("t1→0: best single option (max(C,P))", xy=(t1s[5], max(call, put)),
                fontsize=8, color=ORANGE, xytext=(0, 6), textcoords="offset points")
    ax.annotate("t1→T2: straddle (C+P)", xy=(t1s[5], call + put), fontsize=8,
                color=GREEN, xytext=(0, 6), textcoords="offset points")
    ax.axvline(p["t1"], color=GRAY, linewidth=0.9, linestyle=":")
    style(ax, "Choice-date sensitivity (the paper's unexplored dimension)",
          "choice date $t_1$ (years)", "chooser price")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "t1_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    return {"t1_0.01": vals[0], "t1_paper": cf, "t1_max": vals[-1],
            "max_single": max(call, put), "straddle": call + put}


def rule_gap_figure(p):
    """Cost of the paper's moneyness rule vs interest rate, computed with
    common random numbers (same seed) so the paired gap is nearly noise-free."""
    base = dict(p)
    rs = np.linspace(0.0, 0.10, 21)
    gaps_abs, gaps_bp = [], []
    for r in rs:
        pp = {**base, "r": float(r)}
        o = chooser_mc(**pp, n_paths=400_000, rule="optimal", seed=11)
        m = chooser_mc(**pp, n_paths=400_000, rule="moneyness", seed=11)
        cf = chooser_closed_form(**pp)
        gaps_abs.append(o.price - m.price)
        gaps_bp.append((o.price - m.price) / cf * 1e4)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ax.plot(rs * 100, gaps_bp, color=BLUE, linewidth=2)
    ax.axvline(base["r"] * 100, color=GRAY, linewidth=0.9, linestyle=":")
    ax.annotate("paper's r = 0.15%", xy=(base["r"] * 100, gaps_bp[0]),
                fontsize=8, color=GRAY, xytext=(6, 10), textcoords="offset points")
    style(ax, "Value lost by the paper's moneyness rule vs. rational rule\n"
              "(common-random-number paired estimate)",
          "risk-free rate r (%)", "underpricing (basis points of chooser value)")
    fig.tight_layout()
    fig.savefig(FIG / "rule_gap_vs_r.png", bbox_inches="tight")
    plt.close(fig)
    return {"r_grid": rs.tolist(), "gap_bp": gaps_bp}


def path_table(p, n=10, seed=2021):
    """Paper Table-3-style illustration with both rules shown."""
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    tau = p["T2"] - p["t1"]
    drift = p["r"] - p["q"] - 0.5 * p["sigma"] ** 2
    S1 = p["S0"] * np.exp(drift * p["t1"] + p["sigma"] * np.sqrt(p["t1"]) * z1)
    S2 = S1 * np.exp(drift * tau + p["sigma"] * np.sqrt(tau) * z2)
    rows = []
    for s1, s2 in zip(S1, S2):
        c = bs_call(s1, p["K"], p["r"], p["q"], p["sigma"], tau)
        v = bs_put(s1, p["K"], p["r"], p["q"], p["sigma"], tau)
        rat = "CALL" if c >= v else "PUT"
        mon = "CALL" if s1 > p["K"] else "PUT"
        payoff = max(s2 - p["K"], 0) if rat == "CALL" else max(p["K"] - s2, 0)
        rows.append({"S_t1": round(float(s1), 2), "rational": rat,
                     "moneyness": mon, "agree": rat == mon,
                     "S_T2": round(float(s2), 2), "payoff": round(float(payoff), 2)})
    return rows


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    p = load_params()

    cf = chooser_closed_form(**p)
    cfd = chooser_closed_form_direct(**p)
    call = bs_call(p["S0"], p["K"], p["r"], p["q"], p["sigma"], p["T2"])
    put = bs_put(p["S0"], p["K"], p["r"], p["q"], p["sigma"], p["T2"])
    mc_o = chooser_mc(**p, n_paths=1_000_000, rule="optimal")
    mc_m = chooser_mc(**p, n_paths=1_000_000, rule="moneyness")

    summary = {
        "params": p,
        "closed_form": cf,
        "closed_form_direct": cfd,
        "implementation_agreement": abs(cf - cfd),
        "vanilla_call": call,
        "vanilla_put": put,
        "straddle": call + put,
        "flexibility_premium_over_call": cf - call,
        "discount_vs_straddle": call + put - cf,
        "mc_optimal": [mc_o.price, mc_o.std_error],
        "mc_moneyness": [mc_m.price, mc_m.std_error],
        "rule_gap_paired": mc_o.price - mc_m.price,
        "convergence": convergence_figure(p, cf),
        "t1_sensitivity": t1_figure(p, cf),
        "rule_gap_vs_r": rule_gap_figure(p),
        "path_table": path_table(p),
    }
    sensitivity_figure(p)

    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"closed form           : {cf:.4f}  (direct formula agrees to {abs(cf-cfd):.1e})")
    print(f"call/put/straddle     : {call:.4f} / {put:.4f} / {call+put:.4f}")
    print(f"MC optimal (1M paths) : {mc_o.price:.4f} ± {mc_o.std_error:.4f}")
    print(f"MC moneyness (paired) : {mc_m.price:.4f}  gap = {mc_o.price-mc_m.price:.4f}"
          f" ({(mc_o.price-mc_m.price)/cf*1e4:.1f} bp)")
    g = summary["rule_gap_vs_r"]["gap_bp"]
    print(f"rule gap at r=0%/5%/10%: {g[0]:.1f} / {g[10]:.1f} / {g[20]:.1f} bp")
    agree = sum(r["agree"] for r in summary["path_table"])
    print(f"path table: rational vs moneyness agree on {agree}/10 sample paths")
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
