# MS-Chooser-Option

> **Public snapshot.** This is a single-commit public copy of a private working
> repository (state as of 2026-11-14). Two files are omitted: the internship
> project brief and the PDF of the reference paper (Huang, Wang & Wan 2021,
> *Exploration of JPMorgan Chooser Option Pricing*, BCP Business & Management 15);
> see `references/paper_notes.md` for the citation. The weekly data-refresh
> workflow is set to manual because this copy carries no API-key secrets.
> WRDS / OptionMetrics extracts were never part of the repository.

Advanced chooser option pricing on JPM: BSM replication → real-data baseline →
ML enhancement under a sealed-test protocol → deployable dual-pricing tool.

**Headline results**: closed-form replication $29.1299 (two independent
implementations agree to 7e-15); constant-volatility cost ≈ 35% of price vs an
ex-post oracle (1310 trading days); under a rigorous out-of-sample protocol no
ML configuration beats the EWMA-anchored baseline — hence the tool prices with
**EWMA-anchored BSM as primary** and shows the ML adjustment as a flagged
reference (Weeks 1-8, `docs/week8/final_report_cn.pdf`).

**Weeks 9-12 (enhancement)**: online stack (PostgreSQL + Redis + scheduler + Docker),
point-in-time S&P 500 universe (S&P 1500 scale test), full-stack profiling. Final tool
(Week 12): primary anchor = **GARCH(1,1) 1-day forecast** (EWMA fallback for names with
< 750 returns); ML reference = cross-sectional GBDT on the GARCH anchor for point-in-time
S&P 500 members — its out-of-sample increment over the anchor (2023-24) is significant
cross-sectionally under a pre-registered, size-checked test, conditional on that period.
Implied volatility (OptionMetrics, local research only) is the strongest anchor tested
but is positioned as a research benchmark. Full story: `docs/week12/final_report_cn.pdf`.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in Alpha Vantage / FRED keys
python -m src.collect_all       # pull raw data (or use data/ shipped in repo)
python -m src.preprocess        # clean + 18 features
streamlit run app/streamlit_app.py   # launch the pricing tool
```

The tool offers dual pricing with regime-conditional error bands, extreme
scenarios (vol +50%, rate +200bp, ...), sensitivity curves, archived
performance metrics, and one-click live-data refresh. GitHub Actions refreshes
data every Friday after US close.

## Online stack (Week 9+)

The tool runs as a service: PostgreSQL holds prices, features and daily
quotes, Redis caches hot quotes, and a scheduler refreshes data every trading
day after the US close. One command brings everything up:

```bash
cp .env.example .env            # API keys; DATABASE_URL / REDIS_URL are preset for compose
docker compose up -d --build    # postgres + redis + migrate/backfill + app + scheduler
open http://localhost:8501
```

Without Docker (development machine):

```bash
bash scripts/dev_services.sh start      # local PostgreSQL + Redis (see script header for binaries)
python -m src.db.migrate && python -m src.db.ingest
python -m src.universe.sp500 --load     # S&P 500 point-in-time membership -> securities
python -m src.db.features_db --backfill-garch   # Week 12: GARCH state for an existing database (one-off)
python -m src.service.pipeline          # batched daily run over the universe (add --dry-run to skip network)
streamlit run app/streamlit_app.py
```

If `DATABASE_URL` is unset the service layer reads the `data/` files, so every
Week 1-8 script still works offline. Audits and the performance baseline:
`python analysis/week9/01_online_stack_audits.py`, `python analysis/week9/02_perf_baseline.py`.

## Reproduce the research

| Stage | Command | Audits |
|---|---|---|
| W3 BSM replication | `python -m src.analysis_w3` | `python analysis/week3/01_bsm_chooser_model.py` + `02_model_validation.py` |
| W4 baseline vs oracle | `python -m src.analysis_w4` | `python analysis/week4/01_baseline_evaluation.py` |
| W5 ML frameworks | `python -m src.analysis_w5` | `python analysis/week5/01_ml_frameworks.py` |
| W6 tuning + sealed test + SHAP | `python -m src.analysis_w6` | `python analysis/week6/01_final_evaluation.py` |
| W7 scenarios + tool service | `python -m src.analysis_w7` | `python analysis/week7/01_tool_audits.py` |
| W9 GARCH anchor vs EWMA | `python -m src.analysis_w9` | `python analysis/week9/01_online_stack_audits.py` |
| W10 S&P 500 panel: GARCH + anchored ML round 1 | `python -m src.analysis_w10` | `python analysis/week10/01_universe_audits.py`, `02_perf_after.py` |
| W11 round-2 cross-sectional ML (test opened once), IV anchor | `python -m src.analysis_w11` | `python analysis/week11/01_profile_stack.py --tag after`, `02_perf_audits.py` |
| W12 pre-registered significance (H1-H4), IV on the panel, bands | `python -m src.analysis_w12` (criteria: `docs/week12/preregistration.json`) | `python analysis/week12/01_w12_audits.py` (also re-runs W7, W9-W11) |
| W12 report numbers / claims / final profile | `python analysis/week12/04_report_numbers.py`, `05_claims_check.py` | `python analysis/week12/02_profile_final.py --tag final` |

All runs are seed-fixed and bit-reproducible on the same data version and platform
(data sources revise history; the GARCH optimiser's last digits differ between macOS and
Linux). The OptionMetrics extract (`python -m src.data_collection.wrds_optionm`) needs a
WRDS account and stays on the author's machine under the licence; scripts skip that part
without it.

## Structure

```
app/                 Streamlit tool (thin UI over src/tool/pricing_service.py)
src/db, src/cache, src/service   PostgreSQL / Redis / daily pipeline + scheduler (Week 9)
docker-compose.yml   one-command online stack
src/                 collectors, preprocessing, pricing models, ML, metrics, tool service
analysis/week{N}/    runnable walkthrough + audit scripts (no notebooks, per spec)
data/raw|processed/  datasets (auto-refreshed weekly by GitHub Actions)
docs/week{N}/        per-week written deliverables, figures, result JSONs
reports/             weekly LaTeX reports + cumulative learning report
models/              deployed model bundles (final_*.joblib)
configs/             paper replication parameters
```
