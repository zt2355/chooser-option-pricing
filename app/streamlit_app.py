"""JPM Chooser Option 定价工具 (Week 7-9) — 佟铸城 (Zhucheng Tong)

Run:  streamlit run app/streamlit_app.py
All pricing logic lives in src/tool/pricing_service.py (auditable, UI-free).
Week 9: data comes from PostgreSQL + Redis when configured (falls back to the
Week 2 files otherwise); new "运行状态" tab shows pipeline runs and cache stats.
Week 12: primary anchor = GARCH(1,1) 1-day forecast (EWMA for tickers with < 750
returns); every quote shows which anchor it used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import config
from src.cache import redis_cache as cache
from src.tool.pricing_service import (load_history, load_market_state,
                                      price_chooser, quote, refresh_data,
                                      resolve_source, scenario_table)

st.set_page_config(page_title="Chooser Option 定价工具", page_icon="📈",
                   layout="wide")
st.title("JPM Chooser Option 定价工具")
st.caption("主定价 = GARCH(1,1) 1 日锚定 BSM 闭式解(历史不足 750 个交易日的标的退回 EWMA);"
           "ML 参考价仅作参考。"
           "数据:Yahoo Finance / FRED,每个交易日收盘后自动更新。")

SOURCE = resolve_source()


@st.cache_resource
def _tickers() -> list[str]:
    if SOURCE == "db":
        from src.db import repository as repo
        t = sorted(repo.list_securities()["ticker"].tolist())
        if t:
            return t
    return [config.TICKER]


@st.cache_data(ttl=600)
def _state(ticker: str, as_of: str | None):
    return load_market_state(as_of, ticker=ticker)


@st.cache_data(ttl=600)
def _history(ticker: str):
    df, _ = load_history(ticker)
    return df["jpm_close"].dropna()


with st.sidebar:
    st.header("合约参数")
    _tk = _tickers()
    ticker = st.selectbox("标的", _tk, index=_tk.index(config.TICKER) if config.TICKER in _tk else 0)
    state0 = _state(ticker, None)
    K = st.number_input("行权价 K(默认 ATM)", value=round(state0.S, 2), step=1.0)
    t1 = st.slider("选择日 t₁(年)", 0.05, 0.95, 0.5, 0.05)
    T2 = st.slider("到期日 T₂(年)", max(t1 + 0.05, 0.25), 2.0, 1.0, 0.05)
    st.divider()
    badge = {"db": "🟢 PostgreSQL", "file": "🟡 本地文件"}[SOURCE]
    st.caption(f"数据源:{badge} · 缓存:{'🟢 Redis' if cache.available() else '⚪ 未连接'}")
    if st.button("↻ 拉取最新市场数据"):
        with st.spinner("采集 → 入库 → 特征 → 报价 → 刷新缓存 …"):
            st.success(refresh_data(ticker))
        st.cache_data.clear()

state = _state(ticker, None)
q = quote(state, K=K, t1=t1, T2=T2)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["全市场概览", "定价", "价格趋势", "敏感性与压力", "性能指标", "运行状态"])

with tab0:
    st.subheader("S&P 500 全市场概览(第 10 周)")
    if SOURCE == "db":
        from src.db import repository as repo

        @st.cache_data(ttl=600)
        def _universe():
            return repo.read_universe_quotes()

        uq = _universe()
        if len(uq):
            latest = uq["date"].max()
            uq_day = uq[uq["date"] == latest]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("已定价标的", f"{len(uq_day)}", help=f"as-of {latest.date()}")
            sig_col = "sigma_anchor" if uq_day["sigma_anchor"].notna().any() else "sigma_ewma"
            c2.metric("σ_锚 中位数", f"{uq_day[sig_col].median():.1%}",
                      help="主定价所用波动率锚(GARCH 1 日;历史不足者为 EWMA)")
            c3.metric("主定价/现价 中位数", f"{(uq_day['primary_price'] / uq_day['spot']).median():.1%}",
                      help="ATM 1 年期 chooser 价格占股价的比例")
            c4.metric("恐慌状态占比", f"{(uq_day['regime'] == 'fear').mean():.0%}")
            show = uq_day.reset_index()[["ticker", "name", "sector", "spot", "anchor", "sigma_anchor", "sigma_ewma",
                                         "regime", "primary_price", "band_lo", "band_hi"]]
            show = show.rename(columns={"spot": "S", "anchor": "锚", "sigma_anchor": "σ_锚", "sigma_ewma": "σ_EWMA",
                                        "primary_price": "主定价", "band_lo": "区间下", "band_hi": "区间上"})
            st.dataframe(show.sort_values("主定价", ascending=False).round(4), use_container_width=True, height=420)
            st.caption("全市场只展示主定价(GARCH 1 日锚定 BSM;“锚”列为 ewma 的标的历史不足 750 个交易日)。"
                       "ML 参考价见定价页(适用范围:报价当日的点时 S&P 500 成员且有 GARCH 锚)。"
                       "股息率用于定价时上限 25%(超出者为特别/清算分红)。")
            by_sec = uq_day.groupby("sector").agg(n=("spot", "size"), sigma_med=(sig_col, "median"),
                                                  price_to_spot=("primary_price", lambda x: (x / uq_day.loc[x.index, "spot"]).median()))
            st.caption("按行业:标的数、σ_锚 中位数、主定价/现价中位数")
            st.dataframe(by_sec.round(4), use_container_width=True)
        else:
            st.info("尚无全市场报价,请先运行流水线。")
    else:
        st.info("全市场概览需要数据库(当前为本地文件模式)。")

with tab1:
    a, b, c = st.columns(3)
    anchor_name = {"garch1d": "GARCH 1 日", "ewma": "EWMA"}[q["anchor"]]
    a.metric(f"主定价({anchor_name} 锚定 BSM)", f"${q['primary_price']:.2f}",
             help=f"误差区间 = 该锚在 JPM 上按市场状态的历史定价误差(MAE,第 4 周口径;当前状态:{q['regime']})。"
                  "GARCH 锚的区间校准窗口为 2020-12 至 2024-12,不含 2020 年危机,危机中可能偏窄。")
    a.caption(f"误差区间 ${q['primary_band'][0]:.2f} – ${q['primary_band'][1]:.2f}")
    if q.get("ml_in_scope", True):
        b.metric("ML 参考价", f"${q['ml_reference_price']:.2f}",
                 delta=f"{q['ml_reference_price'] - q['primary_price']:+.2f} vs 主定价")
        b.caption("⚠️ 横截面 GBDT 修正(GARCH 锚);相对锚的增量只在 2023–24 年样本外面板上检验过,仅供参考")
    else:
        b.metric("ML 参考价", "不适用")
        b.caption("适用范围为报价当日的点时 S&P 500 成员且有 GARCH 锚;范围外不做外推。")
    c.metric("市场快照", q["as_of"], help="S/σ/r/q 均取该交易日")
    c.caption(f"S=${q['S']:.2f} · σ_锚={q['sigma_anchor']:.1%} · σ_EWMA={q['sigma_ewma']:.1%} · "
              f"r={q['r']:.2%} · q={q['q']:.2%} · 状态:{q['regime']}")
    st.divider()
    d, e, f_ = st.columns(3)
    d.metric("Call 腿(K, T₂)", f"${q['call_leg']:.2f}")
    e.metric("Put 腿(K, T₂)", f"${q['put_leg']:.2f}")
    f_.metric("Straddle 对照", f"${q['straddle']:.2f}",
              delta=f"chooser 便宜 {q['straddle'] - q['primary_price']:.2f}")

with tab2:
    st.line_chart(_history(ticker).rename(f"{ticker} 收盘价"))
    if SOURCE == "db":
        from src.db import repository as repo
        qh = repo.read_quotes(ticker, limit=250)
        if len(qh):
            st.subheader("每日主定价存档(quotes_daily)")
            st.line_chart(qh[["primary_price", "ml_reference_price"]].sort_index())
    st.caption("历史由调度器每个交易日收盘后自动更新;侧栏按钮可手动拉取最新。")

with tab3:
    st.subheader("极端情景(第 7 周规格:波动率±、利率±、组合压力)")
    st.dataframe(scenario_table(state, t1=t1, T2=T2), use_container_width=True)
    st.subheader("波动率敏感性(当前参数)")
    sig_grid = [state.sigma_anchor * m / 10 for m in range(5, 21)]
    sens = pd.DataFrame({
        "sigma": sig_grid,
        "price": [price_chooser(state.S, K, state.r_1y, state.q, s_, t1, T2)
                  for s_ in sig_grid]}).set_index("sigma")
    st.line_chart(sens)

with tab4:
    st.subheader("基线与模型性能(第 4/6 周评估存档)")
    base = pd.read_csv(config.PROJECT_ROOT / "docs" / "week4" / "baseline_metrics.csv",
                       index_col=0)
    st.caption("第 4 周:回望校准 vs 事后真值(1310 交易日)——8 指标 + 综合分")
    st.dataframe(base.round(3), use_container_width=True)
    final = pd.read_csv(config.PROJECT_ROOT / "docs" / "week6" / "final_comparison.csv",
                        index_col=0)
    st.caption("第 6 周:ML 模型测试集终评(EWMA 基线综合分=1.0,全部 ML>1 → 当时主定价采用 EWMA 锚定;"
               "第 12 周起改为 GARCH 1 日锚,依据见第 11、12 周文档)")
    st.dataframe(final.round(3), use_container_width=True)

with tab5:
    st.subheader("在线运行状态(第 9 周)")
    c1, c2 = st.columns(2)
    c1.metric("数据源", {"db": "PostgreSQL", "file": "本地文件"}[SOURCE])
    c2.metric("Redis 缓存", "已连接" if cache.available() else "未连接")
    if SOURCE == "db":
        from src.db import repository as repo
        counts = repo.table_counts()
        st.caption("表行数:" + " · ".join(f"{k} {v}" for k, v in counts.items()))
        runs = repo.read_runs(15)
        if len(runs):
            st.dataframe(runs[["id", "job", "started_at", "finished_at", "status",
                               "tickers", "rows_prices", "rows_features",
                               "rows_quotes", "data_version", "message"]], use_container_width=True)
    st.json(cache.stats())
