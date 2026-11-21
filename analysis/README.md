# analysis/

按周存放的分析脚本（纯 Python，**不使用 Jupyter Notebook**——导师要求，2026-07-29 起生效）。

每个脚本可从项目根目录直接运行，自带说明文档字符串与分节输出：

```bash
python analysis/week3/01_bsm_chooser_model.py
python analysis/week3/02_model_validation.py
```

| 文件夹 | 内容 |
|---|---|
| `week3/` | BSM chooser 模型实现讲解（01）与模型验证（02） |
| `week9/` | 在线栈八项审计（01）与性能基线测量（02）；需先 `bash scripts/dev_services.sh start` 或 `docker compose up` |
| `week10/` | 股票池与批量流水线十项审计（01，含第 9 周审计回归）、优化前后实验 E1–E5（02）|
| `week11/` | 全流程剖析（01，--tag before/after）与九项性能审计（02，含第 10、9 周回归） |
| `week12/` | 十二项审计（01，含第 11→10→9 周与第 7 周回归）、终测剖析（02）、数据库事实快照（03）、报告数字宏生成与检查（04）、结论清单核对（05） |
