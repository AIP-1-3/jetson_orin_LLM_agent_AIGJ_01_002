#!/usr/bin/env bash
# 1단계 사전 점검 (읽기 전용, sudo 불필요).
# 강의자료는 JetPack 6 기준이라, 우리 장비가 6 인지 7 인지 먼저 판별하고
# 자료의 위험한 명령(SSD 포맷, 스왑 중복 추가)을 건너뛰어도 되는지 알려 준다.
set -u

echo "== 장비 =="
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo

echo "== JetPack 판별 =="
L4T=$(head -n1 /etc/nv_tegra_release 2>/dev/null | sed -n 's/^# R\([0-9]*\).*/\1/p')
UBU=$(lsb_release -rs 2>/dev/null)
PY=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "L4T R${L4T:-?} / Ubuntu ${UBU:-?} / Python ${PY}"
case "${L4T:-0}" in
  36) echo "-> JetPack 6: torch 인덱스 jp6/cu126 (Python 3.10 전용)" ;;
  38|39) echo "-> JetPack 7: torch 인덱스 cu132, torchaudio 는 설치하지 않음 (자료 Q7)" ;;
  *) echo "-> 판별 실패: /etc/nv_tegra_release 를 확인하세요" ;;
esac

echo "== apt 패키지 (02_install_torch.sh 에 필요) =="
MISSING=$(for p in python3-venv python3-pip ffmpeg libsndfile1 libopenblas-dev; do dpkg -s "$p" >/dev/null 2>&1 || echo "$p"; done)
if [ -n "$MISSING" ]; then
  echo "-> 빠짐: $(echo $MISSING). 관리자 계정에서: sudo apt install -y $(echo $MISSING)"
else
  echo "-> 모두 설치됨"
fi

echo "== CUDA =="
if [ -x /usr/local/cuda/bin/nvcc ]; then
  /usr/local/cuda/bin/nvcc --version | tail -n1
  command -v nvcc >/dev/null || echo "주의: nvcc 가 PATH 에 없음 (자료 Q5). torch 는 nvcc 없이도 동작"
else
  echo "/usr/local/cuda 없음 (CUDA 툴킷·cuDNN·TensorRT 미설치). torch cu132 휠은 런타임을 같이 받아 동작 가능. TensorRT 가 필요하면 관리자 계정에서: sudo apt install -y nvidia-jetpack"
fi

echo "== 디스크 =="
ROOT_DEV=$(findmnt -n -o SOURCE /)
echo "루트(/) 장치: $ROOT_DEV"
case "$ROOT_DEV" in
  /dev/nvme*) echo "-> NVMe 로 부팅 중. 자료 1단계 0번(SSD 파티션·포맷)은 실행하지 마세요 (디스크 전체 삭제)" ;;
  *) echo "-> NVMe 부팅 아님. SSD 를 데이터용으로 쓰려면 자료 0번 참고 (새 SSD 일 때만)" ;;
esac
df -h / | tail -n1

echo "== 메모리·스왑 =="
free -h
if swapon --show=NAME --noheadings | grep -q .; then
  echo "-> 스왑 이미 있음. 자료 1단계 3번(스왑 추가)은 건너뛰세요"
else
  echo "-> 스왑 없음. 자료 1단계 3번으로 8G 스왑 추가 권장"
fi

echo "== 전력 모드 =="
nvpmodel -q 2>/dev/null || echo "nvpmodel -q 는 sudo 가 필요할 수 있음: sudo nvpmodel -q"

echo "== HF 캐시 =="
echo "HF_HOME=${HF_HOME:-미설정 (~/.cache/huggingface 사용)}"
