#!/usr/bin/env bash
# 1단계 5번: 실습용 venv(~/eg2) 생성 + JetPack 에 맞는 GPU 용 PyTorch 설치 + 실제 GPU 커널 검증.
# sudo 는 apt 패키지 설치에만 쓴다. 이미 venv 가 있으면 그대로 재사용한다.
set -euo pipefail

VENV="${EG2_VENV:-$HOME/eg2}"
L4T=$(head -n1 /etc/nv_tegra_release | sed -n 's/^# R\([0-9]*\).*/\1/p')

sudo apt install -y python3-venv python3-pip ffmpeg libsndfile1 libopenblas-dev

[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel

case "$L4T" in
  38|39)
    # JetPack 7: 공식 SBSA 휠. torchaudio 는 cu132 용이 없어 sentence-transformers 를 깨뜨리므로 제외 (자료 Q7)
    pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu132
    pip uninstall -y torchaudio 2>/dev/null || true
    ;;
  36)
    pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
    ;;
  *)
    echo "알 수 없는 L4T 버전: R$L4T (setup/01_check_env.sh 로 확인)" >&2
    exit 1
    ;;
esac

# is_available() 만으로는 부족하다. 실제 행렬곱 커널이 돌아야 휠이 맞는 것이다.
python - <<'EOF'
import torch
print("torch", torch.__version__, "capability", torch.cuda.get_device_capability(),
      "bf16", torch.cuda.is_bf16_supported())
x = torch.rand(1024, 1024, device="cuda")
print("matmul sum", (x @ x).sum().item())
EOF

# 이후 설치가 torch 를 PyPI 판으로 덮어쓰지 않도록 버전 고정 파일을 만든다 (자료 2단계)
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > "$VENV/torch-pin.txt"
echo "고정 버전:"; cat "$VENV/torch-pin.txt"
