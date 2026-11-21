# data/

| 子文件夹 | 所属周 | 内容 |
|---|---|---|
| `raw/` | **Week 1** 交付物 | 初始原始数据集（JPM、VIX、S&P 500、国债利率，2018–2024） |
| `processed/` | **Week 2** 交付物 | 清洗对齐后的结构化数据集（≥10 特征，CSV/Parquet） |
| （PostgreSQL） | **Week 9** | 文件数据经 `python -m src.db.ingest` 回填到数据库；此后由每日流水线增量更新，文件与 GitHub Actions 周更保留为回退 |
| `universe/` | **Week 10** | S&P 500 当前成分、历史变更与点时成员区间（Wikipedia 公开来源，可再分发） |
