#!/usr/bin/env bash
# 4단계 자료 만들기: IP 카메라(.env 의 CAM_URL)에서 10초 녹화 + 2초마다 프레임 저장.
# NVIDIA 판 ffmpeg 에 RTSP·libx264 가 없어서 OpenCV 로 녹화한다 (lab/record_clip.py 참고).
# 사용: scripts/record_clip.sh [초]   (기본 10초)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env 없음: cp .env.example .env 후 CAM_URL 을 채우세요" >&2; exit 1; }
MEDIA="${EG2_MEDIA:-media}"
VENV="${EG2_VENV:-$HOME/eg2}"
mkdir -p "$MEDIA"
EG2_MEDIA="$(cd "$MEDIA" && pwd)" "$VENV/bin/python" lab/record_clip.py "${1:-10}"
ls -lh "$MEDIA"/clip.mp4 "$MEDIA"/frame_*.jpg
