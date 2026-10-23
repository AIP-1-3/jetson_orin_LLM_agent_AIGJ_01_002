#!/usr/bin/env bash
# 4단계 자료 만들기: IP 카메라(.env 의 CAM_URL)에서 10초 녹화 + 2초마다 프레임 추출.
# 사용: scripts/record_clip.sh [초]   (기본 10초)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env 없음: cp .env.example .env 후 CAM_URL 을 채우세요" >&2; exit 1; }
set -a; source .env; set +a
MEDIA="${EG2_MEDIA:-media}"
SEC="${1:-10}"
mkdir -p "$MEDIA"
# -loglevel error: 실패 메시지에 계정이 포함될 수 있어 화면 출력은 최소로 둔다
ffmpeg -y -loglevel error -rtsp_transport tcp -i "$CAM_URL" -t "$SEC" -an -c:v libx264 "$MEDIA/clip.mp4"
ffmpeg -y -loglevel error -i "$MEDIA/clip.mp4" -vf fps=0.5 "$MEDIA/frame_%02d.jpg"
ls -lh "$MEDIA"/clip.mp4 "$MEDIA"/frame_*.jpg
ffprobe -v error -show_entries format=duration -of csv=p=0 "$MEDIA/clip.mp4"
