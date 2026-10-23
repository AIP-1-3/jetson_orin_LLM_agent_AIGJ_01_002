#!/usr/bin/env bash
# 2단계: torch 버전을 고정한 채 sentence-transformers 등 설치, 모델 미리 받기.
# 02_install_torch.sh 를 먼저 실행해야 한다.
set -euo pipefail

VENV="${EG2_VENV:-$HOME/eg2}"
PIN="$VENV/torch-pin.txt"
[ -f "$PIN" ] || { echo "$PIN 없음: setup/02_install_torch.sh 먼저 실행" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# torchaudio 는 [audio] extra 가 끌어오지 않도록 extra 없이 설치하고 오디오는 soundfile·librosa 로 읽는다
# opencv-python-headless: venv 가 시스템 OpenCV 를 보지 못하고, 이 휠은 자체 FFmpeg 로 RTSP 를 연다 (4·8·9단계 카메라)
pip install -c "$PIN" -U \
  "sentence-transformers[image,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub opencv-python-headless
if ! grep -q '^torchaudio==' "$PIN"; then
  pip uninstall -y torchaudio 2>/dev/null || true
fi

# 실습 중 다운로드 대기를 없애려고 미리 받는다 (1.5~3GB)
hf download google/embeddinggemma-2

python "$(dirname "$0")/../lab/check.py"
