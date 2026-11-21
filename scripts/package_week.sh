#!/bin/bash
# 按 mentor 要求打包某一周的"代码类"交付物(报告类 PDF 单独发送,不打包)
# 用法: bash scripts/package_week.sh 1   →  生成 dist/week1_deliverables.zip
set -e
N=${1:?用法: bash scripts/package_week.sh <周数>}
cd "$(dirname "$0")/.."

OUT="dist/week${N}_deliverables.zip"
mkdir -p dist
rm -f "$OUT"

# 打包内容:代码、数据、当周文档源文件;排除密钥、内部笔记、报告、git 与缓存
zip -r "$OUT" \
    src data requirements.txt requirements-data.txt README.md \
    $( [ -d configs ] && echo configs ) \
    $( [ -d "docs/week${N}" ] && echo "docs/week${N}" ) \
    $( [ -d "analysis/week${N}" ] && echo "analysis/week${N}" ) \
    $( [ -d models ] && [ "${N}" -ge 5 ] && echo models ) \
    $( [ -d app ] && [ "${N}" -ge 7 ] && echo app ) \
    $( [ "${N}" -ge 9 ] && echo docker-compose.yml Dockerfile .dockerignore scripts/dev_services.sh requirements-data.txt ) \
    -x "*.env*" -x "*__pycache__*" -x "*.ipynb_checkpoints*" \
    -x "*.DS_Store" -x "*HANDOFF*" -x "*/~$*" -x "*wrds_local*" >/dev/null

echo "已生成: $OUT"
unzip -l "$OUT" | tail -3
