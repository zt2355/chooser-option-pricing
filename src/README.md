# src/

代码为累积式（后面每周在此基础上扩展），各文件首次交付所属周如下：

| 文件/模块 | 所属周 |
|---|---|
| `config.py`、`collect_all.py`、`data_collection/`（yahoo / fred / alpha_vantage 采集器） | **Week 1** |
| 预处理与特征工程管道 | Week 2（待建） |
| BSM chooser 定价模型 | Week 3（待建） |
| ML 模型（波动率预测 / 端到端定价） | Week 5–6（待建） |
| Streamlit 定价工具 | Week 7–8（待建） |
| `features.py`（通用特征算术）、`db/`（schema/engine/repository/migrate/ingest）、`cache/redis_cache.py`、`service/`（pipeline/scheduler）、`analysis_w9.py`（GARCH 锚定对照） | **Week 9** |
| `universe/sp500.py`（点时 S&P 500 成员）、`data_collection/multi_source.py`（多源批量采集）、`db/features_db.py`（增量特征+落库状态）、`service/pipeline.py`（批量版）、`tool/pricing_service.py` 的 `quote_universe`/`scenario_universe`、`analysis_w10.py`（面板 GARCH + 锚定 ML 第一轮） | **Week 10** |
| `db/features_db.rebuild_many`、`repository.features_latest`/批量写入、`pipeline` 并发市场请求、`universe/sp1500_scale.py`（压测池）、`analysis_w11.py`（第二轮横截面锚定 ML、SHAP、隐含波动率锚） | **Week 11** |
| `garch_state.py`（带状态 GARCH 锚）、`ml/cross_sectional.py`（部署的横截面 ML 参考模型）、`data_collection/wrds_optionm.py`（OptionMetrics 提取，仅本机）、`analysis_w12.py`（预注册显著性检验、面板隐含波动率、误差带）、`tool/pricing_service.py` 换锚、`service/scheduler.py` 心跳补跑、`universe/sp500.py` 退出后代码停用 | **Week 12** |
