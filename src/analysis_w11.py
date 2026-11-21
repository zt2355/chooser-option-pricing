"""Week 11 theory line.

Part A  Anchored residual learning, ROUND 2, on the S&P 500 panel.
        Round 1 (Week 10) lost everywhere and over-estimated systematically:
        the panel-level label mean drifts between the training and
        validation windows. Round 2 therefore learns only the CROSS-SECTIONAL
        part of the correction:
            y_cs(t,i) = y(t,i) - mean_i y(t,·)          (demeaned by date)
            X_cs      = X z-scored within each date (market-wide level removed)
            + sector dummies (GICS, "Unknown" for names without a sector)
        and prices with sigma_ml = sigma_anchor * exp(clip(y_cs_hat)) — the
        market-level correction is deliberately left at zero (that is what
        the anchor is for). Two anchors are tried: EWMA (Week 4) and the
        GARCH(1,1) 1-day forecast (Week 10 second candidate).
        Protocol unchanged: split by date 70/15/15 with a 252-row purge,
        quarterly walk-forward inside validation for model selection, then
        the TEST period (2024-05-24..2024-12-31) is opened ONCE for the
        configuration chosen on validation plus the two baselines.
Part B  SHAP on the test-period model (feature attribution of the
        cross-sectional correction).
Part C  Implied-volatility anchor (JPM only, OptionMetrics, LOCAL research
        use under the WRDS licence — the extract is git-ignored and nothing
        from it enters the tool or the deliverables): 365-day 50-delta
        implied vol as a third anchor on the Week 4 JPM oracle panel.

Part D  Robustness diagnostics AFTER the single opening (non-selection use):
        rerun without the (look-ahead) sector dummies, GARCH-1d anchor by
        train/validation/test window, level-vs-cross-section split of the
        correction, the daily rank-IC series, and point-in-time guards
        (truncate inputs at the validation end -> earlier values unchanged;
        label windows end before each walk-forward block).

Outputs (docs/week11/): ml_round2_val.csv, ml_round2_test.csv, round2_diagnostics.json,
  ml_round2_summary.json, iv_anchor_jpm.json, figures/ml_round2_test.png,
  figures/shap_round2.png, figures/iv_anchor_jpm.png

Usage: python -m src.analysis_w11
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.analysis_w4 import T1, T2
from src.analysis_w10 import ML_FEATURES, build_panel, price, safe_composite
from src.metrics import composite_score, metric_suite
from src.models.chooser import chooser_closed_form

OUT = config.PROJECT_ROOT / "docs" / "week11"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN, GRAY, PURPLE = "#2f6db3", "#d1603d", "#3a7d44", "#8a8a8a", "#7b4fa3"
INK, GRID = "#333333", "#d9d9d9"
HORIZON, BLOCK, RATIO_CLIP = 252, 63, 1.5
ANCHORS = {"ewma": "sigma_ewma", "garch1d": "sigma_garch_1d"}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK); ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ data

def panel_round2() -> pd.DataFrame:
    from src.db import repository as repo
    p = build_panel()
    g = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed" / "garch_anchors_sp500.parquet")
    g["date"] = pd.to_datetime(g["date"])
    p = p.merge(g[["ticker", "date", "sigma_garch_1d"]], on=["ticker", "date"], how="left")
    sec = repo.list_securities(active_only=False).set_index("ticker")["sector"]
    p["sector"] = p["ticker"].map(sec).fillna("Unknown")
    return p


def cross_sectional(p: pd.DataFrame, anchor_col: str, sector_dummies: bool = True) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Demeaned target and within-date z-scored features (+ sector dummies).
    NB: sectors come from the CURRENT constituent table, so "Unknown" marks names that
    later left the index (look-ahead); Part D reruns everything without the dummies."""
    y = np.log(p["sigma_oracle"] / p[anchor_col])
    y_cs = y - y.groupby(p["date"]).transform("mean")
    X = p[ML_FEATURES].astype(float)
    mu = X.groupby(p["date"]).transform("mean"); sd = X.groupby(p["date"]).transform("std").replace(0, np.nan)
    Xz = ((X - mu) / sd).fillna(0.0).clip(-5, 5)
    if not sector_dummies:
        return Xz, y_cs.to_numpy(float), y
    dummies = pd.get_dummies(p["sector"], prefix="sec", dtype=float)
    Xall = pd.concat([Xz, dummies], axis=1)
    return Xall, y_cs.to_numpy(float), y


def walk_forward(p, Xall, y_cs, dates, period_dates, make_model, standardize=True):
    """Quarterly refits over `period_dates`; train on rows with date <= block start - HORIZON - 1."""
    from sklearn.preprocessing import StandardScaler
    date_pos = {d: i for i, d in enumerate(dates)}
    pred = pd.Series(np.nan, index=p.index)
    blocks = [period_dates[i:i + BLOCK] for i in range(0, len(period_dates), BLOCK)]
    max_train = None
    for blk in blocks:
        cut_pos = date_pos[blk[0]] - HORIZON - 1
        if cut_pos < 0:
            continue
        cut = dates[cut_pos]; max_train = cut if max_train is None else max(max_train, cut)
        tr = (p["date"] <= cut).to_numpy(); te = p["date"].isin(blk).to_numpy()
        Xtr, ytr, Xte = Xall.to_numpy(float)[tr], y_cs[tr], Xall.to_numpy(float)[te]
        model = make_model()
        if standardize:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model.fit(Xtr, ytr)
        pred[te] = np.clip(model.predict(Xte), -RATIO_CLIP, RATIO_CLIP)
    return pred, max_train, model


def evaluate(p_sub: pd.DataFrame, anchor_col: str, pred: pd.Series, label: str) -> tuple[dict, float]:
    s = p_sub.copy()
    s["p_oracle"] = price(s, "sigma_oracle")
    s["p_ref"] = price(s, "sigma_ewma")                    # the ruler is ALWAYS EWMA (Week 4)
    s["sigma_ml"] = np.maximum(s[anchor_col] * np.exp(pred.loc[s.index].to_numpy(float)), 0.01)
    s["p_ml"] = price(s, "sigma_ml")
    ref = metric_suite(s["p_ref"], s["p_oracle"], s["fear_regime"])
    m = metric_suite(s["p_ml"], s["p_oracle"], s["fear_regime"]); m["composite"] = composite_score(m, ref)
    wins = []
    for _, g in s.groupby("ticker"):
        if len(g) < 63:
            continue
        r_ = metric_suite(g["p_ref"], g["p_oracle"], g["fear_regime"])
        c_ = safe_composite(metric_suite(g["p_ml"], g["p_oracle"], g["fear_regime"]), r_)
        if not np.isnan(c_):
            wins.append(c_ < 1)
    return m, float(np.mean(wins)) if wins else float("nan")


LAST: dict = {}          # internals of the reported run, reused by Part D (no effect on outputs)


def part_a_b(sector_dummies: bool = True, write: bool = True) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    p = panel_round2().dropna(subset=["sigma_garch_1d"]).reset_index(drop=True)
    dates = np.array(sorted(p["date"].unique()))
    n = len(dates); i_tr, i_va = int(n * .70), int(n * .85)
    val_dates, test_dates = dates[i_tr:i_va], dates[i_va:]
    split = {"train_end": str(pd.Timestamp(dates[i_tr - 1]).date()), "val_start": str(pd.Timestamp(val_dates[0]).date()),
             "val_end": str(pd.Timestamp(val_dates[-1]).date()), "test_start": str(pd.Timestamp(test_dates[0]).date()),
             "test_end": str(pd.Timestamp(test_dates[-1]).date())}
    p_noTest = p[p["date"] <= val_dates[-1]].reset_index(drop=True)      # SEAL for the selection stage
    models = {
        "ridge_cs": (lambda: Ridge(alpha=10.0), True),
        "gbdt_cs": (lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                                          min_samples_leaf=200, random_state=0), False),
    }
    # ---- validation (selection) ----
    val_rows = []
    preds_val = {}
    for aname, acol in ANCHORS.items():
        Xall, y_cs, _ = cross_sectional(p_noTest, acol, sector_dummies)
        d_no = np.array(sorted(p_noTest["date"].unique()))
        for mname, (mk, std) in models.items():
            pred, max_train, _ = walk_forward(p_noTest, Xall, y_cs, d_no, val_dates, mk, std)
            sub = p_noTest[p_noTest["date"].isin(val_dates) & pred.notna()]
            m, win = evaluate(sub, acol, pred, f"{mname}@{aname}")
            val_rows.append({"config": f"{mname}@{aname}", "anchor": aname, "model": mname, **m, "win_rate": win,
                             "rows": len(sub), "max_train_date": str(pd.Timestamp(max_train).date())})
            preds_val[f"{mname}@{aname}"] = pred
        # anchor itself on validation (no ML): pred = 0
        sub = p_noTest[p_noTest["date"].isin(val_dates)]
        m, win = evaluate(sub, acol, pd.Series(0.0, index=p_noTest.index), aname)
        val_rows.append({"config": f"anchor_only@{aname}", "anchor": aname, "model": "none", **m, "win_rate": win,
                         "rows": len(sub), "max_train_date": None})
    val = pd.DataFrame(val_rows).set_index("config")
    if write:
        val.to_csv(OUT / "ml_round2_val.csv")
    ml_only = val[val["model"] != "none"]
    chosen = ml_only["composite"].idxmin()
    # ---- test: opened ONCE, chosen config + baselines ----
    aname = val.loc[chosen, "anchor"]; mname = val.loc[chosen, "model"]; acol = ANCHORS[aname]
    Xall, y_cs, _ = cross_sectional(p, acol, sector_dummies)
    mk, std = models[mname]
    pred_t, max_train_t, model_t = walk_forward(p, Xall, y_cs, dates, test_dates, mk, std)
    sub_t = p[p["date"].isin(test_dates) & pred_t.notna()]
    test_rows = []
    m, win = evaluate(sub_t, acol, pred_t, chosen)
    test_rows.append({"config": chosen, **m, "win_rate": win, "rows": len(sub_t)})
    for a2, c2 in ANCHORS.items():
        m, win = evaluate(sub_t, c2, pd.Series(0.0, index=p.index), a2)
        test_rows.append({"config": f"anchor_only@{a2}", **m, "win_rate": win, "rows": len(sub_t)})
    test = pd.DataFrame(test_rows).set_index("config")
    if write:
        test.to_csv(OUT / "ml_round2_test.csv")
    # ratio-level diagnostics on test
    y_full = y_cs
    te_mask = sub_t.index
    ss_res = float(((y_full[te_mask] - pred_t.loc[te_mask].to_numpy()) ** 2).sum())
    ss_tot = float(((y_full[te_mask] - y_full[te_mask].mean()) ** 2).sum())
    r2_cs = 1 - ss_res / ss_tot
    ic = float(pd.Series(pred_t.loc[te_mask].to_numpy()).corr(pd.Series(y_full[te_mask]), method="spearman"))
    if not write:                                   # Part D ablation: numbers only, no files
        return {"chosen_on_validation": chosen,
                "validation": {k: float(v) for k, v in val["composite"].items()},
                "test": {k: {"composite": float(r["composite"]), "MAE": float(r["MAE"]), "win_rate": float(r["win_rate"])} for k, r in test.iterrows()},
                "test_rank_IC": ic, "test_ratio_R2_cs": r2_cs}
    LAST.update(p=p, pred_t=pred_t, sub_idx=te_mask, y_cs=y_cs, acol=acol, dates=dates, i_tr=i_tr, i_va=i_va)
    # figure: test composite
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=200)
    names = list(test.index); vals = test["composite"].to_numpy(float)
    ax.barh(names, vals, color=[GREEN if v < 1 else BLUE for v in vals]); ax.axvline(1, color=ORANGE, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    style(ax, f"TEST {split['test_start']}..{split['test_end']} — composite vs EWMA (opened once)", "composite", "")
    fig.tight_layout(); fig.savefig(FIG / "ml_round2_test.png"); plt.close(fig)
    # ---- Part B: SHAP on the last test-block model ----
    shap_out = None
    try:
        import shap
        Xte = Xall.to_numpy(float)[sub_t.index][-4000:]
        if mname == "gbdt_cs":
            ex = shap.TreeExplainer(model_t); sv = ex.shap_values(Xte)
        else:
            ex = shap.LinearExplainer(model_t, Xall.to_numpy(float)[p["date"] <= val_dates[-1]][:20000]); sv = ex.shap_values(Xte)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=Xall.columns).sort_values(ascending=False)
        shap_out = imp.head(12).round(5).to_dict()
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
        ax.barh(imp.head(12).index[::-1], imp.head(12).to_numpy()[::-1], color=BLUE)
        style(ax, f"SHAP mean |value| — {chosen}, test period (cross-sectional correction)", "mean |SHAP| (log-ratio units)", "")
        fig.tight_layout(); fig.savefig(FIG / "shap_round2.png"); plt.close(fig)
    except Exception as e:  # SHAP is diagnostic; never block the result
        shap_out = {"error": str(e)[:200]}
    summary = {"split": split, "sealed_until_selection_done": True,
               "max_train_date_selection": str(ml_only["max_train_date"].max()),
               "max_train_date_test": str(pd.Timestamp(max_train_t).date()),
               "chosen_on_validation": chosen, "validation": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} for k, v in val.to_dict("index").items()},
               "test": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test.to_dict("index").items()},
               "test_ratio_R2_cs": r2_cs, "test_rank_IC": ic, "shap_top": shap_out,
               "panel": {"rows": int(len(p)), "tickers": int(p["ticker"].nunique())}}
    return summary


# ------------------------------------------------------------------ Part C

def part_c() -> dict:
    """JPM: OptionMetrics 365d 50-delta IV as an anchor on the Week 4 oracle panel."""
    from src.analysis_w4 import build_panel as jpm_panel
    path = config.PROJECT_ROOT / "data" / "wrds_local" / "optionm_jpm_surface.parquet"
    if not path.exists():
        return {"skipped": "local OptionMetrics extract not present"}
    iv = pd.read_parquet(path)
    iv365 = iv[(iv["days"] == 365)].pivot_table(index="date", columns="cp_flag", values="impl_volatility")
    iv365["sigma_iv"] = iv365[["C", "P"]].mean(axis=1)
    jp = jpm_panel().join(iv365[["sigma_iv"]], how="inner").dropna(subset=["sigma_iv"])
    cands = {"sigma_ewma": "EWMA", "sigma_iv": "implied 365d", "sigma_252d": "252d realised"}
    prices = {c: pd.Series(np.asarray(chooser_closed_form(S0=jp["jpm_close"].to_numpy(), K=jp["jpm_close"].to_numpy(),
                                                          r=jp["r_1y"].to_numpy(), q=jp["div_yield_ttm"].to_numpy(),
                                                          sigma=jp[c].to_numpy(), t1=T1, T2=T2), float), index=jp.index)
              for c in list(cands) + ["sigma_oracle"]}
    ref = metric_suite(prices["sigma_ewma"], prices["sigma_oracle"], jp["fear_regime"])
    out = {}
    for c in cands:
        m = metric_suite(prices[c], prices["sigma_oracle"], jp["fear_regime"]); m["composite"] = composite_score(m, ref)
        out[c] = {k: float(v) for k, v in m.items()}
    # IV as a FEATURE for the anchored residual: how much of ln(oracle/EWMA) does ln(IV/EWMA) explain?
    y = np.log(jp["sigma_oracle"] / jp["sigma_ewma"]); x = np.log(jp["sigma_iv"] / jp["sigma_ewma"])
    corr = float(pd.Series(y).corr(pd.Series(x)))
    # mechanism check: x and y share -log(EWMA), so their correlation is large even if IV
    # carried no information -> replace IV by a constant and recompute (deterministic).
    # (A circular-shift null was tried and dropped: log IV has AR(1) 0.99 and the shift test
    #  rejected 19% of the time at nominal 5% on independent synthetic series -- mis-sized.)
    le, lo, li = np.log(jp["sigma_ewma"]), np.log(jp["sigma_oracle"]), np.log(jp["sigma_iv"])
    corr_const = float(np.corrcoef(np.log(jp["sigma_iv"].mean()) - le, y)[0, 1])
    def _r2(cols):
        A = np.column_stack([np.ones(len(y))] + cols); b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        return float(1 - ((y.to_numpy() - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    # level vs timing: a CONSTANT anchor at the sample-mean IV (uses the whole sample: diagnostic only)
    # and a point-in-time constant-ish anchor (expanding mean of 252d realised vol)
    jp = jp.assign(const_iv_mean=jp["sigma_iv"].mean(), expanding_rv252=jp["sigma_252d"].expanding().mean())
    diag_anchor = {}
    for c in ("const_iv_mean", "expanding_rv252"):
        pc = pd.Series(np.asarray(chooser_closed_form(S0=jp["jpm_close"].to_numpy(), K=jp["jpm_close"].to_numpy(), r=jp["r_1y"].to_numpy(),
                                                      q=jp["div_yield_ttm"].to_numpy(), sigma=jp[c].to_numpy(), t1=T1, T2=T2), float), index=jp.index)
        m = metric_suite(pc, prices["sigma_oracle"], jp["fear_regime"]); diag_anchor[c] = {"composite": composite_score(m, ref), "MAE": m["MAE"]}
    # same window as the GARCH anchors (GARCH needs 750 observations -> starts later)
    from src.analysis_w9 import garch_anchors
    full = pd.read_csv(config.DATA_PROCESSED / "features_daily.csv", index_col="date", parse_dates=True)
    ov = jp.join(garch_anchors(full["log_return"])[["sigma_garch_1d", "sigma_garch_1y"]], how="inner").dropna(subset=["sigma_garch_1d", "sigma_garch_1y"])
    def _p(c):
        return pd.Series(np.asarray(chooser_closed_form(S0=ov["jpm_close"].to_numpy(), K=ov["jpm_close"].to_numpy(), r=ov["r_1y"].to_numpy(),
                                                        q=ov["div_yield_ttm"].to_numpy(), sigma=ov[c].to_numpy(), t1=T1, T2=T2), float), index=ov.index)
    po_ov = _p("sigma_oracle"); ref_ov = metric_suite(_p("sigma_ewma"), po_ov, ov["fear_regime"])
    same_window = {"start": str(ov.index.min().date()), "end": str(ov.index.max().date()), "days": int(len(ov))}
    for c in ("sigma_ewma", "sigma_iv", "sigma_garch_1d", "sigma_garch_1y"):
        m = metric_suite(_p(c), po_ov, ov["fear_regime"]); same_window[c] = {"composite": composite_score(m, ref_ov), "MAE": m["MAE"]}
    mechanism = {"corr_if_iv_replaced_by_constant": corr_const,
                 "R2_ratio_on_logEWMA": _r2([le.to_numpy()]), "R2_ratio_on_logEWMA_plus_logIV": _r2([le.to_numpy(), li.to_numpy()]),
                 "corr_level_iv_vs_oracle": float(np.corrcoef(jp["sigma_iv"], jp["sigma_oracle"])[0, 1]),
                 "mean_sigma": {c: float(jp[f"sigma_{c}"].mean()) for c in ("iv", "oracle", "ewma")},
                 "std_sigma": {c: float(jp[f"sigma_{c}"].std()) for c in ("iv", "oracle", "ewma")},
                 "anchor_diagnostics": diag_anchor,
                 "same_window_vs_garch": same_window,
                 "reading": "the 0.78 ratio correlation is mechanical (shared -log EWMA); IV's edge is its level, not timing"}
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=200)
    ax.plot(jp.index, jp["sigma_oracle"], color=GRAY, lw=1, label="oracle (fwd 252d realised)")
    ax.plot(jp.index, jp["sigma_ewma"], color=BLUE, lw=1, label="EWMA")
    # the OptionMetrics series itself is NOT drawn: plotting the daily IV path would come close to
    # redistributing licensed data (WRDS terms). Its summary statistics are in iv_anchor_jpm.json.
    style(ax, "JPM: EWMA anchor vs oracle (implied-vol series not drawn: WRDS licence)", "", "annualised sigma")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(FIG / "iv_anchor_jpm.png"); plt.close(fig)
    return {"days": int(len(jp)), "start": str(jp.index.min().date()), "end": str(jp.index.max().date()),
            "metrics": out, "corr_log_iv_ratio_vs_log_oracle_ratio": corr, "mechanism": mechanism,
            "note": "OptionMetrics extract is local-only under the WRDS licence; not in the tool or deliverables"}


# ------------------------------------------------------------------ Part D

def part_d() -> dict:
    """Robustness diagnostics run AFTER the single test opening. Non-selection use:
    nothing here changes the reported configuration or the reported test numbers."""
    p, pred, idx, y_cs = LAST["p"], LAST["pred_t"], LAST["sub_idx"], LAST["y_cs"]
    dates = LAST["dates"]; n = len(dates)
    # 1 sector dummies are look-ahead: "Unknown" = not in TODAY's constituent table
    from src.db import repository as repo
    # "current constituent" is judged from the membership intervals (open interval = still in
    # the index), NOT from the sector field itself (that would be circular)
    mem = repo.read_membership("sp500")
    current = set(mem.loc[mem["end"].isna(), "ticker"])
    unk = p["sector"] == "Unknown"; unk_t = sorted(p.loc[unk, "ticker"].unique())
    sector = {"unknown_row_share": float(unk.mean()), "unknown_tickers": int(len(unk_t)),
              "unknown_later_removed_from_index": int(sum(t not in current for t in unk_t)),
              "unknown_still_in_index": [t for t in unk_t if t in current],
              "ablation_without_sector_dummies": part_a_b(sector_dummies=False, write=False)}
    # 2 GARCH-1d anchor by window. The Week 10 study that nominated this anchor covered
    # 2019-10..2024-12, i.e. it already contained this test window -> pre-test evidence = train/val.
    windows = {"train": dates[:LAST["i_tr"]], "validation": dates[LAST["i_tr"]:LAST["i_va"]], "test": dates[LAST["i_va"]:]}
    anchor = {}
    for k, ds in windows.items():
        s = p[p["date"].isin(ds)]
        po = pd.Series(price(s, "sigma_oracle")); rg = s["fear_regime"].reset_index(drop=True)
        ref = metric_suite(pd.Series(price(s, "sigma_ewma")), po, rg); m = metric_suite(pd.Series(price(s, "sigma_garch_1d")), po, rg)
        anchor[k] = {"rows": int(len(s)), "composite": composite_score(m, ref), "MAE_garch1d": m["MAE"], "MAE_ewma": ref["MAE"]}
    # 3 is the "cross-sectional" correction really cross-sectional, and how stable is it?
    t = p.loc[idx].copy(); t["pred"] = pred.loc[idx].to_numpy(); t["y_cs"] = y_cs[idx]
    daily_mean = t.groupby("date")["pred"].mean()
    t["sigma_level_only"] = t[LAST["acol"]] * np.exp(t["date"].map(daily_mean))
    po = pd.Series(price(t, "sigma_oracle")); rg = t["fear_regime"].reset_index(drop=True)
    ref = metric_suite(pd.Series(price(t, "sigma_ewma")), po, rg)
    lvl = composite_score(metric_suite(pd.Series(price(t, "sigma_level_only")), po, rg), ref)
    t["sigma_cs_only"] = np.maximum(t[LAST["acol"]] * np.exp(t["pred"] - t["date"].map(daily_mean)), 0.01)
    cs_only = composite_score(metric_suite(pd.Series(price(t, "sigma_cs_only")), po, rg), ref)
    ic = t.groupby("date").apply(lambda g: g["pred"].corr(g["y_cs"], method="spearman"))
    stability = {"test_dates": int(len(ic)), "daily_mean_correction": {"min": float(daily_mean.min()), "max": float(daily_mean.max())},
                 "composite_level_shift_only": lvl, "composite_cross_section_only": cs_only, "daily_rank_IC": {"mean": float(ic.mean()), "share_positive": float((ic > 0).mean()),
                                                                      "lag1_autocorr": float(ic.autocorr(1))},
                 "note": "labels look 252 trading days ahead and the test spans ~150 dates: the daily ICs overlap almost fully, so no significance is claimed"}
    # 4 point-in-time guards. (a) cut the raw inputs at D and recompute: every feature, EWMA and
    # GARCH-anchor value dated <= D must be bit-identical. (b) every training row's label window
    # (HORIZON rows of its own ticker ahead) must end before the walk-forward block it feeds.
    from src import features as F
    from src.analysis_w9 import garch_anchors
    from src.db.features_db import _market_inputs, derive_state
    from src.db.engine import get_engine
    import sqlalchemy as sa
    D = pd.Timestamp(windows["validation"][-1]); vix, spx, rates = _market_inputs(); trunc = {}
    for tk in ("JPM", "AAPL", "XOM"):
        eq = repo.read_prices(tk)
        full = derive_state(F.build_features(eq, vix, spx, rates)).loc[:D]
        cut = derive_state(F.build_features(eq.loc[:D], vix.loc[:D], spx.loc[:D], rates.loc[:D])).loc[:D]
        num = [c for c in full.columns if full[c].dtype.kind == "f"]
        lr = repo.read_features(tk)["log_return"].dropna()
        ga = garch_anchors(lr)[["sigma_garch_1d", "sigma_garch_1y"]].loc[:D]
        gb = garch_anchors(lr.loc[:D])[["sigma_garch_1d", "sigma_garch_1y"]].loc[:D]
        trunc[tk] = {"features_identical": bool(np.array_equal(full[num].to_numpy(float), cut[num].to_numpy(float), equal_nan=True)),
                     "garch_identical": bool(np.array_equal(ga.to_numpy(float), gb.to_numpy(float), equal_nan=True)),
                     "rows_checked": int(len(full))}
    with get_engine().connect() as c:
        fd = pd.read_sql(sa.text("SELECT ticker, date FROM features_daily WHERE ticker = ANY(:t) ORDER BY ticker, date"),
                         c, params={"t": sorted(p["ticker"].unique())})
    fd["date"] = pd.to_datetime(fd["date"]); fd["label_end"] = fd.groupby("ticker")["date"].shift(-HORIZON)
    pl = p[["ticker", "date"]].merge(fd, on=["ticker", "date"], how="left")
    gaps = []
    for name, period, pool in (("validation", windows["validation"], pl[pl["date"] <= D]), ("test", windows["test"], pl)):
        dpool = np.array(sorted(pool["date"].unique())); pos = {x: i for i, x in enumerate(dpool)}
        for b0 in range(0, len(period), BLOCK):
            k = pos[period[b0]] - HORIZON - 1
            if k >= 0:
                gaps.append((pd.Timestamp(period[b0]) - pool.loc[pool["date"] <= dpool[k], "label_end"].max()).days)
    pit = {"cut_date": str(D.date()), "truncation": trunc, "label_blocks": len(gaps), "min_label_gap_days": int(min(gaps)),
           "all_pass": bool(all(v["features_identical"] and v["garch_identical"] for v in trunc.values()) and min(gaps) > 0)}
    return {"non_selection_use": True, "sector": sector, "garch1d_anchor_by_window": anchor, "cross_sectional_stability": stability,
            "pit_guards": pit}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = part_a_b()
        print(f"validation (selection): " + ", ".join(f"{k} {v['composite']:.3f}" for k, v in a["validation"].items()))
        print(f"chosen: {a['chosen_on_validation']} | TEST composite: " + ", ".join(f"{k} {v['composite']:.3f} (MAE {v['MAE']:.2f}, win {v['win_rate']:.0%})" for k, v in a["test"].items()))
        print(f"test ratio R2 (cross-sectional) {a['test_ratio_R2_cs']:+.3f}, rank IC {a['test_rank_IC']:+.3f}")
        c = part_c()
        if "metrics" in c:
            print("JPM IV anchor: " + ", ".join(f"{k} {v['composite']:.3f} (MAE {v['MAE']:.2f})" for k, v in c["metrics"].items()) + f" | corr(log IV/EWMA, log oracle/EWMA) {c['corr_log_iv_ratio_vs_log_oracle_ratio']:+.3f}")
    (OUT / "ml_round2_summary.json").write_text(json.dumps(a, indent=2, ensure_ascii=False, default=str))
    (OUT / "iv_anchor_jpm.json").write_text(json.dumps(c, indent=2, ensure_ascii=False, default=str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = part_d()
    (OUT / "round2_diagnostics.json").write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    ab = d["sector"]["ablation_without_sector_dummies"]
    print(f"Part D (non-selection): no sector dummies -> chosen {ab['chosen_on_validation']}, test {ab['test'][ab['chosen_on_validation']]['composite']:.3f}; "
          f"GARCH-1d anchor train/val/test " + "/".join(f"{v['composite']:.3f}" for v in d["garch1d_anchor_by_window"].values()) +
          f"; level-shift only {d['cross_sectional_stability']['composite_level_shift_only']:.3f}; PIT guards {'PASS' if d['pit_guards']['all_pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
