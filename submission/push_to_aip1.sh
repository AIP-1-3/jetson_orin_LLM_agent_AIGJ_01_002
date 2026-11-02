#!/usr/bin/env bash
# 과제 3개를 AIP-1 조직의 제출 저장소로 옮긴다. 맥 터미널에서 실행한다 (Jetson 아님).
# - 커밋 이력을 그대로 옮긴다.
# - 같은 강의자료(EmbeddingGemma 2)를 두 과제로 나눠 내므로, 그 두 저장소는
#   submission/README-<과제>.md 를 README.md 로 바꾼 커밋 하나를 더 얹는다.
# - 대상 저장소에 이미 다른 커밋이 있으면 push 가 거부되어 덮어쓰지 않는다.
# 사용: bash push_to_aip1.sh            (3개 모두)
#       bash push_to_aip1.sh timeseries (하나만: timeseries | embedding_gemma2 | camera_applications)
set -euo pipefail

SRC_TS=https://github.com/AIP-1-3/jenson_AIGJ_01_002.git
SRC_EG=https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002.git
DST=https://github.com/AIP-1
ONLY="${1:-all}"
WORK="$(mktemp -d)"
echo "작업 폴더: $WORK"

# git 사용자 설정이 없으면 GitHub noreply 주소로 커밋한다 (개인 메일을 남기지 않기 위해).
NAME="$(git config --global user.name || true)"
EMAIL="$(git config --global user.email || true)"
NAME="${NAME:-kim75503}"
EMAIL="${EMAIL:-84176988+kim75503@users.noreply.github.com}"

push_main() {  # $1 = 로컬 폴더, $2 = 대상 저장소 주소
  git -C "$1" push "$2" HEAD:main
  echo "완료: ${2%.git}"
}

if [ "$ONLY" = all ] || [ "$ONLY" = timeseries ]; then
  git clone -q "$SRC_TS" "$WORK/timeseries"
  push_main "$WORK/timeseries" "$DST/timeseries-inference-AIGJ_01_002.git"
fi

for name in embedding_gemma2 camera_applications; do
  if [ "$ONLY" != all ] && [ "$ONLY" != "$name" ]; then continue; fi
  case "$name" in
    embedding_gemma2)    dst="$DST/embedding_gemma2-AIGJ_01_002.git";                             scope="1~7단계(환경·임베딩 기초·미세 조정)" ;;
    camera_applications) dst="$DST/practice_task_for_jetson_camera_applications-AIGJ_01_002.git"; scope="8~10단계(카메라 응용)" ;;
  esac
  dir="$WORK/$name"
  git clone -q "$SRC_EG" "$dir"
  cp "$dir/submission/README-$name.md" "$dir/README.md"
  git -C "$dir" rm -r -q submission
  git -C "$dir" add README.md
  git -C "$dir" -c user.name="$NAME" -c user.email="$EMAIL" commit -q \
    -m "제출용 README: 이 과제 범위($scope)만 다루게 정리" \
    -m "같은 강의자료를 두 과제 저장소로 나눠 제출해서, README 를 이 과제 범위의 요약·결과·실행법·셀프 채점으로 바꿈. 코드와 결과 파일은 공통 기반이라 그대로 둠."
  push_main "$dir" "$dst"
done
