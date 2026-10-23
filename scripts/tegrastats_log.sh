#!/usr/bin/env bash
# 측정 중 tegrastats 를 1초 간격으로 results/logs/<이름>_<시각>.log 에 남긴다. Ctrl+C 로 끝낸다.
# 사용: 다른 tmux 창에서 scripts/tegrastats_log.sh 03_text
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?측정 이름을 주세요 (예: 03_text)}"
mkdir -p results/logs
OUT="results/logs/${NAME}_$(date +%Y%m%d-%H%M%S).log"
echo "기록 중: $OUT (Ctrl+C 로 종료)"
# sudo 를 비밀번호 없이 쓸 수 있으면 sudo 로(전체 항목), 아니면 일반 권한으로 실행한다.
# 일반 권한에서도 RAM·GR3D_FREQ(GPU 사용률)는 나온다. 원격 실행처럼 비밀번호를 칠 수 없는 경우를 위한 것
if sudo -n true 2>/dev/null; then TS=(sudo tegrastats); else TS=(tegrastats); fi
"${TS[@]}" --interval 1000 | while IFS= read -r line; do
  printf '%s %s\n' "$(date +%T)" "$line"
done | tee "$OUT"
