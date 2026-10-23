#!/usr/bin/env bash
# 8~10단계: YOLO(ultralytics)와 웹 서버(FastAPI) 설치. sudo 필요 없음.
# 03_install_packages.sh 를 먼저 실행해야 한다.
set -euo pipefail

VENV="${EG2_VENV:-$HOME/eg2}"
PIN="$VENV/torch-pin.txt"
[ -f "$PIN" ] || { echo "$PIN 없음: setup/02_install_torch.sh 먼저 실행" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# torch 가 PyPI 의 일반 aarch64 판(GPU 미지원)으로 바뀌지 않도록 버전을 고정한다
# lap: 추적(track)에 필요. 빠지면 ultralytics 가 첫 실행 중에 자동 설치(AutoUpdate)한다
pip install -c "$PIN" ultralytics lap fastapi "uvicorn[standard]"

# ultralytics 는 opencv-python(GUI 판)을 끌어온다. 이미 쓰는 opencv-python-headless 와
# 같은 cv2 폴더를 덮어써 둘 중 하나가 깨지므로, GUI 판을 지우고 headless 판을 다시 깐다.
if pip show opencv-python >/dev/null 2>&1; then
  pip uninstall -y opencv-python
  pip install -c "$PIN" --force-reinstall --no-deps opencv-python-headless
fi

python - <<'EOF'
import cv2, torch, ultralytics
ffmpeg = [l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith("FFMPEG:")]
print("cv2", cv2.__version__, "|", ffmpeg[0] if ffmpeg else "FFMPEG: 정보 없음")   # RTSP 에 필요
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
EOF
