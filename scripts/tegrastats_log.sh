#!/usr/bin/env bash
# 측정 중 tegrastats 를 1초 간격으로 results/logs/<이름>_<시각>.log 에 남긴다. Ctrl+C 로 끝낸다.
# 사용: 다른 tmux 창에서 scripts/tegrastats_log.sh 03_text
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?측정 이름을 주세요 (예: 03_text)}"
mkdir -p results/logs
OUT="results/logs/${NAME}_$(date +%Y%m%d-%H%M%S).log"
echo "기록 중: $OUT (Ctrl+C 로 종료)"
sudo tegrastats --interval 1000 | while IFS= read -r line; do
  printf '%s %s\n' "$(date +%T)" "$line"
done | tee "$OUT"
