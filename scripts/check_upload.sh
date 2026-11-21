#!/bin/bash
# 检查本地项目与 GitHub 远端是否一致(上传是否成功)
# 用法: bash scripts/check_upload.sh
cd "$(dirname "$0")/.." || exit 1

echo "== 未提交到 git 的本地改动 =="
if [ -z "$(git status --porcelain)" ]; then
  echo "(无,工作区干净)"
else
  git status --short
fi

echo
git fetch origin main --quiet 2>/dev/null || { echo "!! 无法连接 GitHub,请检查网络"; exit 1; }

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

echo "== 本地 vs GitHub =="
if [ "$LOCAL" = "$REMOTE" ]; then
  echo "一致 ✅ GitHub 上已是最新内容 (commit ${LOCAL:0:7})"
else
  AHEAD=$(git rev-list --count origin/main..HEAD)
  BEHIND=$(git rev-list --count HEAD..origin/main)
  [ "$AHEAD" -gt 0 ] && echo "本地领先 $AHEAD 个 commit —— 尚未上传 ❌"
  [ "$BEHIND" -gt 0 ] && echo "远端领先 $BEHIND 个 commit —— GitHub 上有本地没有的内容(例如网页端直接上传),建议先 git pull"
fi
