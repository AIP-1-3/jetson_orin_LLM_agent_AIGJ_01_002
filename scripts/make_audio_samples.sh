#!/usr/bin/env bash
# 5단계 자료 만들기: espeak-ng 로 합성 음성을 만들고, 경보음·잡음 생성과 16 kHz 모노 변환은 python 으로 한다.
# (NVIDIA 판 ffmpeg 에 lavfi 가 없어서. lab/make_audio_samples.py 참고)
set -euo pipefail
cd "$(dirname "$0")/.."
MEDIA="${EG2_MEDIA:-media}"
VENV="${EG2_VENV:-$HOME/eg2}"
mkdir -p "$MEDIA"
command -v espeak-ng >/dev/null || { echo "espeak-ng 없음: 관리자 계정에서 sudo apt install -y espeak-ng" >&2; exit 1; }
espeak-ng -v ko "강의실이 너무 덥습니다. 에어컨을 켜 주세요." -w "$MEDIA/raw_speech.wav"
# 이전 실행에서 ffmpeg 로 만든 raw 파일이 남아 있으면 지워서 python 이 만든 파일과 섞이지 않게 한다
rm -f "$MEDIA/raw_beep.wav" "$MEDIA/raw_noise.wav"
EG2_MEDIA="$(cd "$MEDIA" && pwd)" "$VENV/bin/python" lab/make_audio_samples.py
