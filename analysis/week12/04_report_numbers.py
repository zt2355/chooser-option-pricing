"""Report numbers as LaTeX macros (Week 12) — 佟铸城 (Zhucheng Tong)

Every number the Weeks 9-12 chapters of the final report cite is generated here from
the analysis outputs (JSON/CSV under docs/week9..week12) and written to
docs/week12/numbers.tex as \\newcommand{\\rnName}{value}. The report \\input{}s that
file and never types a Week 9-12 number by hand, so a number cannot drift from the
output that produced it. docs/week12/numbers_sources.csv lists, for each macro, the
file and field it comes from.

  --check   regenerate in memory and fail if docs/week12/numbers.tex is stale, if the
            report uses an undefined \\rn macro, or if a defined macro is never used

Usage: python analysis/week12/04_report_numbers.py [--check]
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs"
REPORTS = [D / "week12" / "final_report_cn.tex", ROOT / "reports" / "week12" / "weekly_report_week12.tex",
           ROOT / "reports" / "learning_report" / "learning_report.tex"]


def J(path):
    return json.load(open(D / path))


def f(x, nd=2):
    return f"{x:.{nd}f}"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def build() -> list[tuple[str, str, str]]:
    w9 = J("week9/perf_baseline.json")["results"]
    w10 = J("week10/panel_summary.json")
    w11 = J("week11/ml_round2_summary.json"); w11d = J("week11/round2_diagnostics.json")
    p11b, p11a = J("week11/profile_before.json"), J("week11/profile_after.json")
    iv11 = J("week11/iv_anchor_jpm.json")
    sig, ivp, bands = J("week12/significance.json"), J("week12/iv_panel.json"), J("week12/bands.json")
    pf, snap = J("week12/profile_final.json"), J("week12/db_snapshot.json")
    perf10 = {r["experiment"]: r for r in csv.DictReader(open(D / "week10/perf_before_after.csv"))}
    T = sig["tests"]; A = sig["part_a"]; R = snap["cited_runs"]; cov = snap["coverage"]
    rows = []

    def add(name, value, source):
        assert re.fullmatch(r"rn[A-Za-z]+", name), name
        rows.append((name, str(value), source))

    # ---- Week 9
    add("rnWnineQuoteMs", f(w9["S5_quote_w9_path_model_resident"]["median_ms"], 1), "week9/perf_baseline.json results.S5.median_ms")
    add("rnWnineRedisMs", f(w9["S6_quote_redis_hit"]["median_ms"], 2), "week9/perf_baseline.json results.S6.median_ms")
    add("rnWnineRenderMs", f(w9["S11_streamlit_first_render"]["median_ms"], 0), "week9/perf_baseline.json results.S11.median_ms")
    add("rnWnineDailyS", f(w9["S10_daily_pipeline_with_network"]["median_ms"] / 1000, 1), "week9/perf_baseline.json results.S10.median_ms/1000")
    # ---- Week 10
    add("rnMemIntervals", cov["membership_intervals"], "week12/db_snapshot.json coverage.membership_intervals")
    add("rnMemTickers", cov["membership_tickers"], "week12/db_snapshot.json coverage.membership_tickers")
    add("rnPanelTenRows", thou(w10["panel"]["rows"]), "week10/panel_summary.json panel.rows")
    add("rnPanelTenTickers", w10["panel"]["tickers"], "week10/panel_summary.json panel.tickers")
    for key, name in (("E1_feature_rebuild_jpm", "rnEone"), ("E2_market_state_load", "rnEtwo"), ("E3_universe_quotes", "rnEthree"),
                      ("E4_universe_scenarios", "rnEfour"), ("E5_daily_pipeline_universe", "rnEfive")):
        add(name, f(float(perf10[key]["speedup"]), 1).rstrip("0").rstrip("."), f"week10/perf_before_after.csv {key}.speedup")
    add("rnTenDailyS", f(R["w10_590_before"]["total_s"], 1), "week12/db_snapshot.json cited_runs.w10_590_before.total_s")
    add("rnGoneDWin", pct(w10["garch"]["g1d_win_rate"]), "week10/panel_summary.json garch.g1d_win_rate")
    add("rnGoneDPooled", f(w10["garch"]["pooled"]["p_g1d"]["composite"], 3), "week10/panel_summary.json garch.pooled.p_g1d.composite")
    m10 = w10["ml"]["metrics"]
    add("rnRoundOneRidge", f(m10["ridge_anch"]["composite"], 2), "week10/panel_summary.json ml.metrics.ridge_anch.composite")
    add("rnRoundOneGbdt", f(m10["gbdt_anch"]["composite"], 2), "week10/panel_summary.json ml.metrics.gbdt_anch.composite")
    add("rnNoPrices", cov["members_without_prices"], "week12/db_snapshot.json coverage.members_without_prices")
    add("rnInStudy", cov["members_in_study_window"], "week12/db_snapshot.json coverage.members_in_study_window")
    add("rnTruncated", len(cov["active_members_truncated_history"]), "week12/db_snapshot.json coverage.active_members_truncated_history")
    add("rnPostExit", len(cov["post_exit_inactive"]), "week12/db_snapshot.json coverage.post_exit_inactive")
    # ---- Week 11 performance
    add("rnLatestBeforeMs", f(p11b["L3_latest_row_query"]["execution_ms"], 0), "week11/profile_before.json L3_latest_row_query.execution_ms")
    add("rnLatestBeforeRows", p11b["L3_latest_row_query"]["rows"], "week11/profile_before.json L3_latest_row_query.rows")
    add("rnLatestFinalMs", f(pf["L3_features_latest_explain"]["execution_ms"], 1), "week12/profile_final.json L3_features_latest_explain.execution_ms")
    add("rnLatestFinalRows", pf["L3_features_latest_explain"]["rows"], "week12/profile_final.json L3_features_latest_explain.rows")
    add("rnQuBeforeMs", f(p11b["L2_quote_universe"]["median_ms"], 0), "week11/profile_before.json L2_quote_universe.median_ms")
    add("rnQuBeforeN", p11b["L2_quote_universe"]["tickers"], "week11/profile_before.json L2_quote_universe.tickers")
    add("rnQuAfterMs", f(p11a["L2_quote_universe"]["median_ms"], 1), "week11/profile_after.json L2_quote_universe.median_ms")
    add("rnQuAfterN", p11a["L2_quote_universe"]["tickers"], "week11/profile_after.json L2_quote_universe.tickers")
    add("rnMarketBeforeS", f(R["w10_590_before"]["stages"]["market"], 1), "week12/db_snapshot.json cited_runs.w10_590_before.stages.market")
    add("rnMarketAfterS", f(R["w11_590_after"]["stages"]["market"], 1), "week12/db_snapshot.json cited_runs.w11_590_after.stages.market")
    add("rnFeatBeforeS", f(R["w10_590_before"]["stages"]["features"], 1), "week12/db_snapshot.json cited_runs.w10_590_before.stages.features")
    add("rnFeatAfterS", f(R["w11_590_after"]["stages"]["features"], 1), "week12/db_snapshot.json cited_runs.w11_590_after.stages.features")
    add("rnDailyAfterS", f(R["w11_590_after"]["total_s"], 1), "week12/db_snapshot.json cited_runs.w11_590_after.total_s")
    add("rnScaleSteadyS", f(R["w11_1517_steady"]["total_s"], 1), "week12/db_snapshot.json cited_runs.w11_1517_steady.total_s")
    add("rnScaleN", R["w11_1517_steady"]["tickers"], "week12/db_snapshot.json cited_runs.w11_1517_steady.tickers")
    add("rnScaleBackfillS", f(R["w11_1517_backfill"]["total_s"], 0), "week12/db_snapshot.json cited_runs.w11_1517_backfill.total_s")
    add("rnRenderBeforeMs", f(p11b["L1_dashboard_first_render"]["median_ms"], 0), "week11/profile_before.json L1_dashboard_first_render.median_ms")
    add("rnRenderAfterMs", f(p11a["L1_dashboard_first_render"]["median_ms"], 0), "week11/profile_after.json L1_dashboard_first_render.median_ms")
    # ---- Week 12 performance (final re-test)
    add("rnFinalQuMs", f(pf["L2_quote_universe"]["median_ms"], 0), "week12/profile_final.json L2_quote_universe.median_ms")
    add("rnFinalQuN", pf["L2_quote_universe"]["tickers"], "week12/profile_final.json L2_quote_universe.tickers")
    add("rnFinalScalarMs", f(pf["L2_quote_scalar_uncached_JPM"]["median_ms"], 0), "week12/profile_final.json L2_quote_scalar_uncached_JPM.median_ms")
    add("rnFinalXsMs", f(pf["L2_ml_cross_section"]["median_ms"], 0), "week12/profile_final.json L2_ml_cross_section.median_ms")
    add("rnFinalRenderMs", f(pf["L1_dashboard_first_render"]["median_ms"], 0), "week12/profile_final.json L1_dashboard_first_render.median_ms")
    g = pf["L4b_garch_one_new_day_simulated"]
    add("rnGarchOrdS", f(g["ordinary_day"]["total_s"], 1), "week12/profile_final.json L4b...ordinary_day.total_s")
    add("rnGarchRefitS", f(g["refit_day"]["total_s"], 1), "week12/profile_final.json L4b...refit_day.total_s")
    add("rnGarchRefits", thou(g["refit_day"]["refits"]), "week12/profile_final.json L4b...refit_day.refits")
    add("rnGarchTickers", thou(g["refit_day"]["tickers"]), "week12/profile_final.json L4b...refit_day.tickers")
    add("rnCatchupS", f(R["w12_1510_catchup_6days"]["total_s"], 1), "week12/db_snapshot.json cited_runs.w12_1510_catchup_6days.total_s")
    add("rnCatchupFeatS", f(R["w12_1510_catchup_6days"]["stages"]["features"], 1), "week12/db_snapshot.json cited_runs.w12_1510_catchup_6days.stages.features")
    add("rnDbFinalMb", f(pf["L3_db_size_mb"], 0), "week12/profile_final.json L3_db_size_mb")
    add("rnDbElevenMb", f(p11a["L3_db_size_mb"], 0), "week11/profile_after.json L3_db_size_mb")
    add("rnRssFinalMb", f(pf["rss_mb"]["after_dashboard"], 0), "week12/profile_final.json rss_mb.after_dashboard")
    add("rnRssElevenMb", f(p11a["rss_mb"]["after_dashboard"], 0), "week11/profile_after.json rss_mb.after_dashboard")
    # ---- Week 11 round 2
    te = w11["test"]
    add("rnElevenTestMl", f(te["gbdt_cs@garch1d"]["composite"], 3), "week11/ml_round2_summary.json test.gbdt_cs@garch1d.composite")
    add("rnElevenTestG", f(te["anchor_only@garch1d"]["composite"], 3), "week11/ml_round2_summary.json test.anchor_only@garch1d.composite")
    add("rnElevenMaeMl", f(te["gbdt_cs@garch1d"]["MAE"], 2), "week11/ml_round2_summary.json test.gbdt_cs@garch1d.MAE")
    add("rnElevenMaeG", f(te["anchor_only@garch1d"]["MAE"], 2), "week11/ml_round2_summary.json test.anchor_only@garch1d.MAE")
    add("rnElevenMaeE", f(te["anchor_only@ewma"]["MAE"], 2), "week11/ml_round2_summary.json test.anchor_only@ewma.MAE")
    add("rnElevenVal", f(w11["validation"]["gbdt_cs@garch1d"]["composite"], 3), "week11/ml_round2_summary.json validation.gbdt_cs@garch1d.composite")
    add("rnElevenTestStart", w11["split"]["test_start"], "week11/ml_round2_summary.json split.test_start")
    add("rnNoDummyTest", f(w11d["sector"]["ablation_without_sector_dummies"]["test"]["gbdt_cs@garch1d"]["composite"], 3),
        "week11/round2_diagnostics.json sector.ablation_without_sector_dummies.test.gbdt_cs@garch1d.composite")
    add("rnUnknownN", w11d["sector"]["unknown_tickers"], "week11/round2_diagnostics.json sector.unknown_tickers")
    add("rnUnknownRemoved", w11d["sector"]["unknown_later_removed_from_index"], "week11/round2_diagnostics.json sector.unknown_later_removed_from_index")
    aw = w11d["garch1d_anchor_by_window"]
    add("rnGTrain", f(aw["train"]["composite"], 3), "week11/round2_diagnostics.json garch1d_anchor_by_window.train.composite")
    add("rnGVal", f(aw["validation"]["composite"], 3), "week11/round2_diagnostics.json garch1d_anchor_by_window.validation.composite")
    add("rnCsOnly", f(w11d["cross_sectional_stability"]["composite_cross_section_only"], 3), "week11/round2_diagnostics.json cross_sectional_stability.composite_cross_section_only")
    add("rnLevelOnly", f(w11d["cross_sectional_stability"]["composite_level_shift_only"], 3), "week11/round2_diagnostics.json cross_sectional_stability.composite_level_shift_only")
    # ---- Week 12 significance (pre-registered)
    add("rnOosStart", A["oos_start"], "week12/significance.json part_a.oos_start")
    add("rnOosEnd", A["oos_end"], "week12/significance.json part_a.oos_end")
    add("rnOosDates", A["oos_dates"], "week12/significance.json part_a.oos_dates")
    add("rnOosRows", thou(A["oos_rows"]), "week12/significance.json part_a.oos_rows")
    add("rnOosTickers", A["tickers"], "week12/significance.json part_a.tickers")
    add("rnOosMaeE", f(A["mae"]["ewma"], 2), "week12/significance.json part_a.mae.ewma")
    add("rnOosMaeG", f(A["mae"]["garch1d"], 2), "week12/significance.json part_a.mae.garch1d")
    add("rnOosMaeMl", f(A["mae"]["ml_nodummies"], 2), "week12/significance.json part_a.mae.ml_nodummies")
    words = {"H1": "One", "H2": "Two", "H3": "Three", "H4": "Four"}
    for h, w in words.items():
        t = T[h]
        add(f"rnH{w}Delta", f(t["delta_mae"], 2 if h != "H2" else 3), f"week12/significance.json tests.{h}.delta_mae")
        add(f"rnH{w}Lo", f(t["ci95_used"][0], 2 if h != "H2" else 3), f"week12/significance.json tests.{h}.ci95_used[0]")
        add(f"rnH{w}Hi", f(t["ci95_used"][1], 2 if h != "H2" else 3), f"week12/significance.json tests.{h}.ci95_used[1]")
        add(f"rnH{w}SizeTicker", pct(t["size_check"]["rejection_rate_at_5pct"], 0), f"week12/significance.json tests.{h}.size_check.rejection_rate_at_5pct")
        fb = t.get("fallback_sector_bootstrap", {}).get("size_check", {})
        if fb:
            add(f"rnH{w}SizeSector", pct(fb["rejection_rate_at_5pct"], 1), f"week12/significance.json tests.{h}.fallback_sector_bootstrap.size_check.rejection_rate_at_5pct")
        add(f"rnH{w}Pholm", f(t["p_holm"], 3), f"week12/significance.json tests.{h}.p_holm")
    by2 = T["H2"]["by_year"]
    add("rnHTwoYThree", f(by2["2023"]["delta_mae"], 3), "week12/significance.json tests.H2.by_year.2023.delta_mae")
    add("rnHTwoYFour", f(by2["2024"]["delta_mae"], 3), "week12/significance.json tests.H2.by_year.2024.delta_mae")
    add("rnHTwoBDelta", f(sig["secondary"]["H2b"]["delta_mae"], 3), "week12/significance.json secondary.H2b.delta_mae")
    st = sig["size_check_self_test"]
    add("rnSelfIidTicker", pct(st["iid_noise"]["ticker_bootstrap_size"], 1), "week12/significance.json size_check_self_test.iid_noise.ticker_bootstrap_size")
    add("rnSelfShockTicker", pct(st["strong_sector_shocks"]["ticker_bootstrap_size"], 0), "week12/significance.json size_check_self_test.strong_sector_shocks.ticker_bootstrap_size")
    add("rnSelfShockSector", pct(st["strong_sector_shocks"]["sector_bootstrap_size"], 1), "week12/significance.json size_check_self_test.strong_sector_shocks.sector_bootstrap_size")
    # ---- IV
    add("rnIvJpm", f(iv11["metrics"]["sigma_iv"]["composite"], 3), "week11/iv_anchor_jpm.json metrics.sigma_iv.composite")
    add("rnIvJpmCorr", f(iv11["corr_log_iv_ratio_vs_log_oracle_ratio"], 2), "week11/iv_anchor_jpm.json corr_log_iv_ratio_vs_log_oracle_ratio")
    add("rnIvJpmCorrConst", f(iv11["mechanism"]["corr_if_iv_replaced_by_constant"], 2), "week11/iv_anchor_jpm.json mechanism.corr_if_iv_replaced_by_constant")
    add("rnIvPanelRows", thou(ivp["rows"]), "week12/iv_panel.json rows")
    add("rnIvPanelTickers", ivp["tickers"], "week12/iv_panel.json tickers")
    for k, w in (("iv", "Iv"), ("garch1d", "Gd"), ("garch1y", "Gy"), ("rv", "Rv"), ("ewma", "Ew")):
        kk = "rv252" if k == "rv" else k
        add(f"rnIvMae{w}", f(ivp["mae"][kk], 2), f"week12/iv_panel.json mae.{kk}")
    add("rnIvWinGd", pct(ivp["iv_ticker_win_rate_vs"]["garch1d"]), "week12/iv_panel.json iv_ticker_win_rate_vs.garch1d")
    add("rnIvWinGy", pct(ivp["iv_ticker_win_rate_vs"]["garch1y"]), "week12/iv_panel.json iv_ticker_win_rate_vs.garch1y")
    mc = ivp["mechanism_descriptive"]
    add("rnIvConstIv", f(mc["mae_iv_constant_at_ticker_mean"], 2), "week12/iv_panel.json mechanism_descriptive.mae_iv_constant_at_ticker_mean")
    add("rnIvConstGy", f(mc["mae_garch1y_constant_at_ticker_mean"], 2), "week12/iv_panel.json mechanism_descriptive.mae_garch1y_constant_at_ticker_mean")
    add("rnIvXsIv", f(mc["xs_corr_ticker_mean_iv_vs_oracle"], 2), "week12/iv_panel.json mechanism_descriptive.xs_corr_ticker_mean_iv_vs_oracle")
    add("rnIvXsGy", f(mc["xs_corr_ticker_mean_garch1y_vs_oracle"], 2), "week12/iv_panel.json mechanism_descriptive.xs_corr_ticker_mean_garch1y_vs_oracle")
    add("rnIvTsIv", f(mc["median_ts_corr_iv_vs_oracle"], 2), "week12/iv_panel.json mechanism_descriptive.median_ts_corr_iv_vs_oracle")
    add("rnIvGapActual", f(ivp["mae"]["garch1y"] - ivp["mae"]["iv"], 2), "week12/iv_panel.json mae.garch1y - mae.iv")
    add("rnIvGapConst", f(mc["mae_garch1y_constant_at_ticker_mean"] - mc["mae_iv_constant_at_ticker_mean"], 2), "week12/iv_panel.json mechanism const gap")
    # ---- bands / tool
    add("rnBandGCalm", f(bands["garch1d"]["calm"], 2), "week12/bands.json garch1d.calm")
    add("rnBandGFear", f(bands["garch1d"]["fear"], 2), "week12/bands.json garch1d.fear")
    add("rnBandESameCalm", f(bands["ewma_same_window_as_garch"]["calm"], 2), "week12/bands.json ewma_same_window_as_garch.calm")
    add("rnBandESameFear", f(bands["ewma_same_window_as_garch"]["fear"], 2), "week12/bands.json ewma_same_window_as_garch.fear")
    add("rnBandEFullCalm", f(bands["ewma_full_w4_panel"]["calm"], 2), "week12/bands.json ewma_full_w4_panel.calm")
    add("rnBandEFullFear", f(bands["ewma_full_w4_panel"]["fear"], 2), "week12/bands.json ewma_full_w4_panel.fear")
    add("rnBandGDays", bands["garch1d"]["days"], "week12/bands.json garch1d.days")
    tool = snap["tool"]
    add("rnToolAsOf", tool["as_of"], "week12/db_snapshot.json tool.as_of")
    add("rnToolQuoted", thou(tool["quoted"]), "week12/db_snapshot.json tool.quoted")
    add("rnToolGarch", thou(tool["anchor_counts"].get("garch1d", 0)), "week12/db_snapshot.json tool.anchor_counts.garch1d")
    add("rnToolEwma", tool["anchor_counts"].get("ewma", 0), "week12/db_snapshot.json tool.anchor_counts.ewma")
    add("rnToolMl", tool["ml_reference_count"], "week12/db_snapshot.json tool.ml_reference_count")
    return rows


def render(rows) -> tuple[str, str]:
    tex = "% generated by analysis/week12/04_report_numbers.py - do not edit by hand\n" + "".join(
        f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v, _ in rows)
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(["macro", "value", "source"]); w.writerows(rows)
    return tex, buf.getvalue()


def main() -> int:
    rows = build()
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)), "duplicate macro"
    tex, src = render(rows)
    out_tex, out_csv = D / "week12" / "numbers.tex", D / "week12" / "numbers_sources.csv"
    if "--check" in sys.argv:
        assert out_tex.read_text() == tex, "numbers.tex is stale: rerun without --check"
        used = set().union(*(re.findall(r"\\(rn[A-Za-z]+)", p.read_text()) for p in REPORTS))
        missing, unused = used - set(names), set(names) - used
        assert not missing, f"undefined macros used in the report: {sorted(missing)}"
        # Chinese LaTeX: a straight double quote renders as a closing quote on both sides
        for p in REPORTS:
            body = "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("%"))
            assert '"' not in body, f"straight double quote in {p.name}: use “ ”"
        print(f"numbers.tex up to date ({len(names)} macros); used {len(used)}; defined but unused: {sorted(unused)}")
        return 0
    out_tex.write_text(tex); out_csv.write_text(src)
    print(f"wrote {len(rows)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
