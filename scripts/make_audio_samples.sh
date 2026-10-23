#!/usr/bin/env bash
# 5단계 자료 만들기: 마이크 없이 합성 음성·경보음·잡음을 만들고 16 kHz 모노로 변환.
set -euo pipefail
cd "$(dirname "$0")/.."
MEDIA="${EG2_MEDIA:-media}"
mkdir -p "$MEDIA"
command -v espeak-ng >/dev/null || sudo apt install -y espeak-ng
espeak-ng -v ko "강의실이 너무 덥습니다. 에어컨을 켜 주세요." -w "$MEDIA/raw_speech.wav"
ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=1000:duration=3" "$MEDIA/raw_beep.wav"
ffmpeg -y -loglevel error -f lavfi -i "anoisesrc=color=pink:duration=5" "$MEDIA/raw_noise.wav"
for f in "$MEDIA"/raw_*.wav; do
  name=$(basename "$f"); name=${name#raw_}
  ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 "$MEDIA/$name"
  printf '%s: ' "$name"; ffprobe -v error -show_entries stream=sample_rate,channels -of csv=p=0 "$MEDIA/$name"
done
