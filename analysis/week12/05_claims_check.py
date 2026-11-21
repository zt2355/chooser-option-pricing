"""Claims checklist for the final report (Week 12) — 佟铸城 (Zhucheng Tong)

Each conclusion-type sentence of the Weeks 9-12 chapters is listed with (a) a text
fragment that must appear in the compiled report source, (b) the output fields it
rests on, (c) an executable check of the claim against those fields, and (d) the
rigor checks it went through. Numbers themselves come from numbers.tex (script 04);
this file checks that the WORDS around them are supported.

Usage: python analysis/week12/05_claims_check.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs"
TEX = (D / "week12" / "final_report_cn.tex").read_text()
WEEKLY = (ROOT / "reports" / "week12" / "weekly_report_week12.tex").read_text()


def J(p):
    return json.load(open(D / p))


sig, ivp, bands = J("week12/significance.json"), J("week12/iv_panel.json"), J("week12/bands.json")
w11d, snap, pf = J("week11/round2_diagnostics.json"), J("week12/db_snapshot.json"), J("week12/profile_final.json")
T = sig["tests"]
qual = lambda h: {y: v for y, v in T[h]["by_year"].items() if v["dates"] >= 120}

CLAIMS = [
    ("ML increment significant, conditional on the period, not over time",
     "以这段时期为条件", "significance.json tests.H2, decisions",
     lambda: sig["decisions"]["ml_increment"] == "ESTABLISHED" and T["H2"]["ci95_used"][0] > 0
     and sorted(qual("H2")) == ["2023", "2024"],
     "pre-registered; size-checked procedure; Holm; identification boundary stated"),
    ("the increment comes mainly from 2024",
     "主要来自 2024 年", "significance.json tests.H2.by_year",
     lambda: qual("H2")["2024"]["delta_mae"] > 5 * qual("H2")["2023"]["delta_mae"] > 0, "descriptive"),
    ("ticker bootstrap mis-sized on all four hypotheses; intervals from the sector procedure",
     "四项假设上按股票的程序全部严重失准", "significance.json tests.*.size_check",
     lambda: all(t["size_check"]["rejection_rate_at_5pct"] > 0.075 for t in T.values())
     and all(t["procedure"] != "ticker bootstrap" for t in T.values()), "size check, self-tested"),
    ("H1 and H4 only descriptive; H2 and H3 significant",
     "仅描述", "significance.json tests.*.procedure / significant_holm_5pct",
     lambda: [T[h]["significant_holm_5pct"] for h in ("H1", "H2", "H3", "H4")] == [False, True, True, False], "pre-registered rule"),
    ("size check self-test: noise ok, strong sector shocks flagged",
     "被正确识别为失准", "significance.json size_check_self_test",
     lambda: 0.025 <= sig["size_check_self_test"]["iid_noise"]["ticker_bootstrap_size"] <= 0.075
     and sig["size_check_self_test"]["strong_sector_shocks"]["ticker_bootstrap_size"] > 0.3, "known-answer test"),
    ("anchor swap improvement about three times the ML increment",
     "约为 ML 增量的三倍", "significance.json tests.H1/H2.delta_mae",
     lambda: 2.5 < T["H1"]["delta_mae"] / T["H2"]["delta_mae"] < 3.5, "same OOS rows"),
    ("anchor swap positive in every qualifying year",
     "且在每个合格年份都为正", "significance.json tests.H1.by_year",
     lambda: all(v["delta_mae"] > 0 for v in qual("H1").values()), "descriptive"),
    ("GARCH anchor beats EWMA in train, validation and test windows",
     "三个窗口都优于 EWMA", "round2_diagnostics.json garch1d_anchor_by_window",
     lambda: all(w11d["garch1d_anchor_by_window"][k]["composite"] < 1 for k in ("train", "validation", "test")), "pre-test windows"),
    ("IV the strongest anchor tested on the panel",
     "检验过的最强的锚", "iv_panel.json mae",
     lambda: min(ivp["mae"], key=ivp["mae"].get) == "iv", "common rows"),
    ("IV advantage is level placement, not timing",
     "而不是择时", "iv_panel.json mechanism_descriptive",
     lambda: abs((ivp["mechanism_descriptive"]["mae_garch1y_constant_at_ticker_mean"] - ivp["mechanism_descriptive"]["mae_iv_constant_at_ticker_mean"])
                 - (ivp["mae"]["garch1y"] - ivp["mae"]["iv"])) < 0.1
     and ivp["mechanism_descriptive"]["median_ts_corr_iv_vs_oracle"] < 0.2
     and ivp["mechanism_descriptive"]["xs_corr_ticker_mean_iv_vs_oracle"] > ivp["mechanism_descriptive"]["xs_corr_ticker_mean_garch1y_vs_oracle"],
     "mechanism decomposition (look-ahead diagnostic, labelled)"),
    ("IV positioned as research benchmark, not candidate",
     "研究基准，不列为候选锚", "significance.json decisions.iv_class",
     lambda: sig["decisions"]["iv_class"] == "research_benchmark", "pre-registered classes"),
    ("band narrowing mostly from the window, not the anchor",
     "区间变窄主要来自窗口而非换锚", "bands.json",
     lambda: all((bands["ewma_full_w4_panel"][r] - bands["ewma_same_window_as_garch"][r])
                 > (bands["ewma_same_window_as_garch"][r] - bands["garch1d"][r]) for r in ("calm", "fear")), "same-window comparison"),
    ("removing sector dummies did not lower performance",
     "结果没有被抬高", "round2_diagnostics.json sector.ablation",
     lambda: w11d["sector"]["ablation_without_sector_dummies"]["test"]["gbdt_cs@garch1d"]["composite"] <= 0.755 + 1e-9, "ablation (non-selection)"),
    ("all of the ML gain is cross-sectional",
     "收益全部来自横截面", "round2_diagnostics.json cross_sectional_stability",
     lambda: w11d["cross_sectional_stability"]["composite_cross_section_only"] < 0.755 < 0.791 < w11d["cross_sectional_stability"]["composite_level_shift_only"],
     "complete two-part decomposition"),
    ("W10 attribution corrected: no current member with truncated history",
     "复核后更正为退市与代码复用", "db_snapshot.json coverage",
     lambda: snap["coverage"]["current_members_truncated"] == [] and len(snap["coverage"]["post_exit_inactive"]) > 0, "independent source (membership intervals)"),
    ("every quote carries its anchor (counts add up)",
     "每笔报价都标明所用的锚", "db_snapshot.json tool",
     lambda: sum(snap["tool"]["anchor_counts"].values()) == snap["tool"]["quoted"], "count identity"),
    ("ordinary-day GARCH step cheaper than refit day",
     "普通交易日", "profile_final.json L4b",
     lambda: pf["L4b_garch_one_new_day_simulated"]["ordinary_day"]["total_s"] < pf["L4b_garch_one_new_day_simulated"]["refit_day"]["total_s"], "simulation labelled"),
    ("latest-state read, database side, like-for-like faster",
     "数据库执行", "week11 profile_before vs week12 profile_final",
     lambda: pf["L3_features_latest_explain"]["execution_ms"] < J("week11/profile_before.json")["L3_latest_row_query"]["execution_ms"], "same measure (EXPLAIN execution)"),
]


def main() -> int:
    bad = 0
    for name, frag, source, check, tests in CLAIMS:
        in_text = frag in TEX or frag in WEEKLY
        ok = bool(check())
        status = "OK " if (in_text and ok) else "BAD"
        bad += status == "BAD"
        print(f"{status} {name}  [text: {'found' if in_text else 'MISSING'}; data: {'supports' if ok else 'CONTRADICTS'}; {source}; {tests}]")
    print(f"\n{len(CLAIMS) - bad}/{len(CLAIMS)} claims supported")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
