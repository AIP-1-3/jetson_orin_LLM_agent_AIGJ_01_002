# Jetson Orin LLM 에이전트 실습: EmbeddingGemma 2 멀티모달 임베딩

Jetson Orin Nano Super 8GB 에서 EmbeddingGemma 2 로 텍스트·이미지·음성을 한 벡터 공간에서 검색하고, IP 카메라 물체 검색·동작 인식으로 넓힌 실습 저장소입니다.

- **해결하는 문제**: 강의자료는 JetPack 6·USB 웹캠 기준이라 우리 장비(JetPack 7, IP 카메라)에서 그대로 돌지 않습니다. 장비 차이를 자동으로 판별해 설치하고, 모든 측정값을 측정 조건과 함께 파일로 남깁니다.
- **핵심 결과** (2호기, MAXN_SUPER, bfloat16): 텍스트 검색 6/6 정답·질의 2개 151 ms, 묶음 32 로 처리량 22배(6.5 → 145.9 질의/초), 도메인 미세 조정 accuracy@1 0.833 → 0.850, 물체 검출 10.9~12.9 fps, COCO 밖 물체(세탁기 사진) 검색 1위 정답 yolo 1 → relabel 2 → yoloe 3개(5개 중), 사람 목소리 녹음으로 음성 검색 3/3, 같은 대본 동작 인식에서 규칙 smooth 가 만세 6/6·손흔들기 3/3(검증 1.000 인 GRU 보다 정확), 학습에 없는 사람도 만세 6/6, 자세 인식 12.9~17.3 fps.
- **시연 화면** (8단계, 2호기 yoloe, 2026-10-26): 사람 얼굴과 주소창이 없는 화면만 올렸습니다.
  - [세탁기 검출 화면](docs/images/08-demo-washer-yoloe.png): 화면에 띄운 세탁기 이미지 검색 결과를 카메라로 비추자 드럼 세탁기 두 대를 "drum washing machine 0.95·0.79" 로 잡은 실시간 영상(6.9 fps).
  - [검색창과 최근 추출 목록](docs/images/08-demo-search.png): 잘라 낸 물체 12개에 임베딩이 고른 한국어 라벨(컵, 드럼 세탁기, 노트북, 마우스, 휴대폰, 키보드)과 YOLO 라벨이 함께 보이는 화면(검출 17.1 fps, 임베딩 606 ms/개). 같은 시연에서 "세탁기"·"컵"·"노트북" 검색은 상위 3개가 모두 정답이었습니다([표](docs/02-results.md#8단계-웹캠-물체-추출과-검색)).
- **문서**: [단계별 결과](docs/02-results.md) · [기술적 고민과 문제 해결 과정](docs/03-engineering-notes.md) · [강의자료 대비 점검과 셀프 채점](docs/01-self-grade.md) · [강의자료 정리](docs/00-lecture-summary.md)

## 단계별 결과 한눈에

| 단계 | 결과 | 근거 |
| --- | --- | --- |
| 1~2 설치 | torch 2.14.1+cu132 (JetPack 7 자동 판별), sentence-transformers 6.2.0 | `setup/`, [문제 해결 1·8](docs/03-engineering-notes.md) |
| 3 텍스트 | 6/6 정답, 150.6~151.9 ms (768/256/128 차원), 초당 13.3 질의. 프롬프트를 빼면 3/6 | [`results/03_text.jsonl`](results/03_text.jsonl) |
| 4 이미지·동영상 | 장면별 6장 1/3 정답, 6장 2.7~2.9 초, 동영상 21초(1 fps 프레임) 9.1 초, 텍스트+이미지 혼합 548 ms | [`results/04_vision.jsonl`](results/04_vision.jsonl) |
| 5 음성 | 합성 음성 2/3 → 사람 목소리 녹음을 넣자 3/3, 문구를 바꾼 질의 3개도 모두 1위(원인은 합성 음성), WAV 4개 379~435 ms | [`results/05_audio.jsonl`](results/05_audio.jsonl), [문제 해결 4](docs/03-engineering-notes.md) |
| 6 성능 | 모델 증가분 텍스트 579 MB · 오디오 948 MB · 전체 1270 MB(한 인덱스 검색 3/6), 묶음 32 로 처리량 22배·질의당 에너지 1/12 | [6단계](docs/02-results.md#6단계-성능-측정) |
| 7 미세 조정 | accuracy@1 0.833 → 0.850, 1 epoch 74.8 초, 일반 질의 24/24 → 21/24 | [`results/07_finetune.jsonl`](results/07_finetune.jsonl) |
| 8 물체 검색 | 같은 물체 5종 통제 실험(각 150초): 검색 1위 정답 yolo 1 · relabel 2 · yoloe 3개, fps 12.9 · 11.5 · 10.9 → 강의실용으로 yoloe 선택. relabel 은 tv → monitor 교정, 대신 맞는 cup 도 물병으로 바꿈 | [`results/08_objects.jsonl`](results/08_objects.jsonl), [8-4 표](docs/02-results.md#8단계-웹캠-물체-추출과-검색) |
| 9 동작 인식 | 같은 대본(만세 6·한팔들기 3·손흔들기 3)에서 smooth 만세 6·손흔들기 3 정확, raw 는 손 흔들기 판정 없음, GRU 는 검증 1.000 이지만 16번 더 셈 → 규칙 smooth 선택. 관심도 4개 상황 6/12(머무름 0/3 원인 분석). 처음 보는 사람(팀원 A): 두 방식 모두 만세 6/6, GRU 손흔들기 1/3. 악수 3회 모두 검출(카운트는 12로 부풂). 12.9~17.3 fps, RAM 3.5~4.0 GB | [`results/09_pose.jsonl`](results/09_pose.jsonl), [9단계](docs/02-results.md#9단계-사람-동작-인식) |
| 10 두 카메라 접근 감지 (진행 중) | 렌즈 보정 RMS 0.063(1호기)·1.81(2호기), 바닥 보드 위치 보정 두 번 차이 0.1cm. ankle1 이 자로 잰 0.52m 를 0.52m 로 측정(25.8 fps). 2m·3m·stereo·fused 는 공간을 넓혀 다시 잼 | [`results/10_*.jsonl`](results/), [10단계](docs/02-results.md#10단계-두-카메라-접근-감지-진행-중-2026-10-27-1일차), [문제 해결 24~30](docs/03-engineering-notes.md) |

강의자료 요구 37개 중 36개를 실측으로 채웠습니다. 남은 칸(8단계 선택 확장 yoloe+relabel)과 셀프 채점은 [docs/01-self-grade.md](docs/01-self-grade.md) 에 있습니다.

## 구성

```mermaid
flowchart LR
  subgraph setup[setup/ 설치]
    A[01_check_env.sh<br/>JetPack·디스크·스왑 판별] --> B[02_install_torch.sh<br/>GPU torch + 커널 검증]
    B --> C[03_install_packages.sh<br/>torch 고정 후 패키지·모델]
  end
  subgraph lab[lab/ 실습]
    E[eg2.py 공통 로더] --> T3[test_text.py]
    E --> T4[test_vision.py]
    E --> T5[test_audio.py]
    CAM[camera.py<br/>USB·RTSP 공통 입력]
    BEN[bench.py<br/>측정값 + 조건 기록]
  end
  C --> E
  T3 & T4 & T5 --> BEN --> R[(results/*.jsonl)]
  TS[scripts/tegrastats_log.sh] --> L[(results/logs/)]
  R & L --> TAB[results_table.py<br/>6단계 표]
```

| 경로 | 역할 |
| --- | --- |
| `setup/01_check_env.sh` | JetPack 6/7, NVMe 부팅 여부, 스왑, 전력 모드를 읽기만 해서 자료의 어느 명령을 건너뛸지 알려 줌 |
| `setup/02_install_torch.sh` | `~/eg2` venv, JetPack 에 맞는 torch 설치, 실제 GPU 행렬곱으로 검증, 버전 고정 파일 생성 |
| `setup/03_install_packages.sh` | torch 고정 상태로 sentence-transformers 6.1+·opencv-python-headless 설치, 모델 미리 받기, `lab/check.py` 실행 |
| `lab/eg2.py` | 필요한 인코더만 골라 올리는 로더 (bfloat16) |
| `lab/test_text.py` `test_vision.py` `test_audio.py` | 3·4·5단계. 데우기 후 두 번째 실행 지연을 재고 정답 여부를 기록 |
| `lab/bench.py` | 지연, 로드 전후 RAM, torch 할당량, 측정 조건(전력 모드·L4T·버전·커밋)을 `results/<단계>.jsonl` 에 추가 |
| `lab/test_all.py` | 5단계 확장·6단계 '전체 (740M)' 행. 이미지·음성·동영상 프레임을 한 인덱스에 넣고 한 질의로 검색 |
| `lab/results_table.py` | `results/` 와 tegrastats 로그로 6단계 표를 Markdown 으로 출력 |
| `lab/camera.py` | USB 번호와 RTSP 주소를 같은 방식으로 열고, 최신 프레임만 유지해 RTSP 지연 누적을 막음 |
| `scripts/record_clip.sh` | IP 카메라에서 10초 녹화 + 2초마다 프레임 저장 (4단계 자료, `lab/record_clip.py` 호출) |
| `lab/record_clip.py` | OpenCV 로 RTSP 를 받아 `clip.mp4`(mp4v)와 `frame_NN.jpg` 저장, 받은 fps 출력 |
| `scripts/make_audio_samples.sh` | 합성 음성·경보음·잡음을 16 kHz 모노로 생성 (5단계 자료) |
| `scripts/tegrastats_log.sh` | 측정 중 tegrastats 를 시각과 함께 `results/logs/` 에 저장 |

## 실행 방법

아래 명령은 모두 **Jetson 에 ssh 접속한 창**에서 실행합니다. 처음 한 번만 저장소를 받습니다.

```bash
git clone https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002.git ~/eg2-lab
```

```bash
cd ~/eg2-lab
```

### 1~2단계 설치

```bash
./setup/01_check_env.sh
```

출력에서 "NVMe 로 부팅 중"이면 자료의 SSD 포맷 명령을, "스왑 이미 있음"이면 스왑 추가를 건너뜁니다.

```bash
./setup/02_install_torch.sh
```

마지막에 `capability (8, 7)`, `bf16 True`, 숫자 하나가 나오면 성공입니다. `no kernel image` 가 나오면 휠이 장비와 맞지 않는 것입니다.

```bash
./setup/03_install_packages.sh
```

`torch ... cuda True` 와 sentence-transformers 6.1 이상이 출력되면 성공입니다.

### 3~5단계 테스트

측정할 때는 tmux 창을 둘로 나눠 한쪽에서 tegrastats 를 기록합니다.

```bash
./scripts/tegrastats_log.sh 03_text
```

다른 창에서:

```bash
source ~/eg2/bin/activate
```

```bash
cd ~/eg2-lab/lab && python test_text.py
```

```bash
python test_text.py --no-prompt
```

`--no-prompt` 는 자료의 변형 과제입니다. 질의·문서 프롬프트를 빼고 같은 측정을 해 점수 차이를 봅니다.

GPU 사용률과 처리량을 보려면 tegrastats 를 짧은 간격으로 켜고 연속 추론합니다.

```bash
TS_MS=200 ./scripts/tegrastats_log.sh 03_text
```

```bash
python test_text.py --sustain 10
```

4단계는 IP 카메라 주소를 `.env` 에 넣은 뒤 자료를 만듭니다.

```bash
cd ~/eg2-lab && cp .env.example .env && nano .env
```

```bash
./scripts/record_clip.sh
```

```bash
cd lab && python test_vision.py --judge Y Y Y
```

`--judge` 는 질의별 1위 이미지를 눈으로 보고 맞으면 Y, 틀리면 N 으로 적습니다. 장면별 녹화 파일(`empty_`·`people_`·`board_`)이면 생략해도 자동 판정합니다.

동영상은 torchcodec 가 없어 `{"video": ...}` 입력이 실패합니다(문제 해결 기록 9). 아래처럼 OpenCV 로 1 fps 프레임을 뽑아 넣고, `--mixed` 로 텍스트+이미지 혼합 입력(확장 과제)도 함께 잽니다.

```bash
python test_vision.py --video-frames --mixed
```

```bash
cd ~/eg2-lab && ./scripts/make_audio_samples.sh
```

```bash
cd lab && python test_audio.py
```

직접 녹음한 목소리(16 kHz 모노)를 `media/` 에 `voice.wav` 로 넣었다면, 질의와 정답 파일을 바꿔 파일별 점수를 모두 남길 수 있습니다(5단계 재확인).

```bash
python test_audio.py --query "에어컨 좀 켜 달라는 말=voice.wav" --query "사람이 말하는 소리=voice.wav,speech.wav"
```

전체 모델(740M)로 이미지·음성·동영상 프레임을 한 인덱스에 넣는 확장 과제이자 6단계 표의 '전체' 행입니다. 가장 무거우니 데스크톱 브라우저를 닫고 `free -h` 로 여유를 본 뒤 실행합니다.

```bash
python test_all.py
```

### 6단계 성능 측정

```bash
python results_table.py
```


결과 표와 해석은 [docs/02-results.md](docs/02-results.md#6단계-성능-측정) 에 있습니다. 숫자는 `results/` 파일에서만 가져옵니다.

### 7단계 미세 조정

```bash
python eval_ir.py
```

```bash
EPOCHS=1 OUT_NAME=eg2-classroom-e1 nohup python train_ft.py > ../results/logs/train.log 2>&1 &
```

```bash
python eval_ir.py ../models/eg2-classroom-e1
```

```bash
python check_general.py ../models/eg2-classroom-e1
```

학습은 메모리가 빠듯해(문제 해결 기록 11) 데스크톱 브라우저를 닫고 tmux 안에서 실행합니다. 첫 명령은 기준 모델 평가, 마지막은 일반 질의 성능 회귀 확인입니다.

### 8~9단계 카메라 응용

자료는 USB 웹캠(`cv2.VideoCapture(0)`)을 가정하지만, 2호기는 IP 카메라라서 입력을 `lab/camera.py` 의 `LatestFrame` 으로 바꿨습니다. 같은 코드가 1호기 USB 카메라(`--source 0`)에서도 돕니다. 검출이 카메라보다 느려도 최신 프레임만 처리해 지연이 쌓이지 않고, 같은 프레임을 두 번 판정하지 않습니다. `--seconds N` 을 주면 N초 뒤 끝내면서 fps·라벨·동작 횟수를 `results/08_objects.jsonl`·`results/09_pose.jsonl` 에 남깁니다.

```bash
bash setup/04_install_detect.sh
```

```bash
cd lab && python app.py --mode yolo --seconds 120
```

8-1 의 검색 확인은 `--probe` 로 남깁니다. 끝날 때 각 질의의 상위 3개(라벨, YOLO 클래스, 점수)가 결과 파일에 들어갑니다. `--min-score 0.3` 은 확장 과제의 점수 필터입니다.

```bash
python app.py --mode relabel --seconds 150 --probe 세탁기 건전지 전지 컵 휴대폰 노트북
```

```bash
cd lab && python pose_app.py --raw --seconds 120
```

9-3 녹화 → 9-6 학습 → GRU 판정, 9-5 관심도 순서입니다. 녹화는 라벨마다 하나씩(idle·만세·한팔들기·손흔들기) 30초 이상 받습니다.

```bash
python pose_app.py --record ../data/action/만세.csv
```

```bash
python train_pose.py --data ../data/action --task action --window 30 --out models/action.pt
```

```bash
python pose_app.py --model models/action.pt --seconds 120
```

```bash
python pose_app.py --behavior --seconds 360
```

브라우저에서 `http://<2호기 Tailscale 주소>:8000` 으로 실시간 영상과 검색 화면을 봅니다. 사람은 크롭·저장하지 않습니다(8단계 `SKIP_CLASSES`). 9단계는 관절 좌표만 다룹니다.

8·9단계 결과(강의자료 8-4·9-4 표 포함)는 [docs/02-results.md](docs/02-results.md#8단계-웹캠-물체-추출과-검색) 에 있습니다.

### 10단계 두 카메라 접근 감지 (진행 중)

cam0 은 2호기 USB 카메라, cam1 은 1호기 USB 카메라입니다. 1호기에서 영상을 내보내고(포트 8140), 2호기에서 두 영상을 받아 계산합니다. 아래 명령은 2호기 `~/eg2-lab/prox` 에서 실행합니다(1호기 명령만 1호기에서). 2호기 카메라도 같은 서버로 8141 에 띄워 맥에서 화면을 확인합니다.

1호기:

```bash
python prox/cam_server.py --cam 0 --port 8140
```

2호기:

```bash
python cam_server.py --cam 0 --port 8141
```

렌즈 보정(카메라마다, 보드를 들고 여러 거리·각도로):

```bash
python calib_intrinsics.py --name cam0 --src http://127.0.0.1:8141/stream.mjpg --square 0.02
```

위치 보정(보드를 두 화면 모두에서 화면 폭 1/6 이상 보이게 바닥에 놓고):

```bash
python calib_extrinsics_board.py --cams http://127.0.0.1:8141/stream.mjpg http://<1호기 Tailscale 주소>:8140/stream.mjpg --square 0.02 --board-height 0
```

접근 감지와 10-7 측정(사람이 줄자로 잰 자리에 선 동안 10초):

```bash
python proximity_app.py --cams http://127.0.0.1:8141/stream.mjpg http://<1호기 Tailscale 주소>:8140/stream.mjpg --mode ankle1 --target-xy 0.08,-0.05
```

```bash
python measure_10_7.py --label "1.0m, 서 있음" --true-d 1.0 --seconds 10
```

10단계 결과와 실패 기록은 [docs/02-results.md](docs/02-results.md#10단계-두-카메라-접근-감지-진행-중-2026-10-27-1일차) 와 문제 해결 24~30 에 있습니다.

## 진행 기록과 회고

| 날짜 | 한 일 | 근거 |
| --- | --- | --- |
| 2026-10-23 | 강의자료 정리, 장비 차이 분석 | [PR #1](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/1) |
| 2026-10-23 | 1~6단계 설치·테스트·측정 코드 (장비 실행 전) | [PR #2](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/2) |
| 2026-10-23 | 2호기 첫 실행: 환경 점검 통과, torch 설치에서 sudo 로 멈춤 → 스크립트 보강 | [PR #4](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/4), 문제 해결 기록 1 |
| 2026-10-23 | 2호기 1~3단계 완료 (torch 2.14.1+cu132, 텍스트 검색 6/6 정답). 다른 프로세스 RAM 이 섞여 모델 증가분 열 추가, tegrastats 를 sudo 없이도 기록 | [PR #5](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/5) |
| 2026-10-23 | 음성 샘플 생성을 파이썬으로 교체, 연속 추론 처리량 측정 추가 | [PR #6](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/6), 문제 해결 기록 2·3 |
| 2026-10-23 | 2호기 3·5단계 측정 결과와 tegrastats 로그 커밋, 표에 전력·처리량 추가 | [PR #7](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/7), 6단계 표, 문제 해결 기록 4 |
| 2026-10-23 | IP 카메라 녹화를 ffmpeg 에서 OpenCV 로 교체 (카메라 VIGI-C540-W 확인, 계정 입력 전) | [PR #8](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/8), 문제 해결 기록 5 |
| 2026-10-23 | 카메라 연결 시험: 대역이 달라 연결 실패, 계정 가리기 버그 발견·수정 | [PR #9](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/9), 문제 해결 기록 6·7 |
| 2026-10-23 | 핫스팟으로 카메라 연결, 4단계 이미지 측정 2회(28 fps 수신, 5장 2.5~2.9 초, 최대 20.2 W), 설치 스크립트에 OpenCV 추가 | [PR #10](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/10), 6단계 표, 문제 해결 기록 7~9 |
| 2026-10-23 | 묶음 크기 처리량 실험·7단계 평가/학습 스크립트, 장면별 녹화 옵션, 이미지 4장씩 나눠 넣기 | [PR #11](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/11) |
| 2026-10-23 | 4단계 3차 측정(10장 5.9 초, RAM 97%) 결과와 로그 커밋 | [PR #12](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/12), 문제 해결 기록 10 |
| 2026-10-23 | 이미지 정답을 장면 파일 이름으로 자동 판정 | [PR #13](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/13) |
| 2026-10-23 | 장면별 6장 측정(1/3 정답, 4장씩 넣어 최대 RAM 87%), 7단계 학습·평가 데이터(AI 작성) | [PR #14](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/14), [data/README.md](data/README.md) |
| 2026-10-23 | 묶음 크기 실험(묶음 32 에서 처리량 22배), 8·9단계 코드를 IP 카메라용으로 작성 (장비 실행 전) | [PR #15](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/15), 묶음 크기 실험 표 |
| 2026-10-23 | 7단계 기준 평가(acc@1 0.833), 학습 OOM 2회 → 임베딩 표 고정·길이 제한 추가 | [PR #16](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/16), 문제 해결 기록 11 |
| 2026-10-23 | 7단계 학습 성공(acc@1 0.833 → 0.850, 일반 질의 6/6 → 3/6), 학습 tegrastats 로그, 일반 성능 확인 스크립트 | [PR #18](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/18) |
| 2026-10-23 | 7단계 1 epoch 비교(acc@1 0.850 유지, 학습 74.8 초), 세 모델 일반 성능 24/24·21/24·21/24 | [PR #19](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/19) |
| 2026-10-23 | 8단계 2호기 실행(yolo 11.8 fps, relabel 이 tv → monitor 교정), 종료 시 exit 134 수정 | [PR #20](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/20), 문제 해결 기록 12 |
| 2026-10-23 | 8단계 yoloe(12.7 fps, 크롭 0), 9단계 자세 2회(13.6~13.8 fps, 손흔들기 23), 손 흔들기 깜빡임 수정, 대용량 모델 파일 git 제외 | [PR #21](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/21), 문제 해결 기록 13 |
| 2026-10-23 | 9단계 재측정(만세 5, 깜빡임 되돌아감 5 → 1), 8단계 물건 배치 재측정(4종 11개, teddy bear → mouse) | [PR #22](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/22) |
| 2026-10-23 | 10단계 1·2호기 Tailscale 연결(접근 규칙 추가), 영상 서버 포트 8091 로 변경 | [PR #23](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/23), 문제 해결 기록 14 |
| 2026-10-26 | 강의자료 요구사항 37개 대비 점검(충족 19 · 부분 6 · 미실시 12)과 루브릭 A 셀프 채점(79 → 84). 8-4·9-4 를 자료 표 형식으로 정리, 동영상 프레임 대체·혼합 입력·전체 모델·검색 기록 옵션 추가, 결과 파일에서 카메라 IP 제거 | [docs/01-self-grade.md](docs/01-self-grade.md), [PR #24](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/24) |
| 2026-10-26 | 마감(10/27) 맞춰 이틀 일정 수립, README 를 요약·실행법으로 줄이고 결과·문제 해결을 docs/02·03 으로 분리, 7단계 평가 데이터 사람 검수표(20쌍), 7단계 실행 명령 추가 | [docs/01-self-grade.md 5절](docs/01-self-grade.md), [docs/04-eval-review.md](docs/04-eval-review.md), 이 PR |
| 2026-10-26 | 2호기 오전 재측정: GPU 확인 로그, 3단계 프롬프트 없음 3/6, 동영상 21초 9.1 초, 혼합 입력, 전체 모델 30개 13.4 초·3/6. 2호기에 GitHub 인증이 없어 결과 커밋을 git bundle 로 옮김. 결과표가 옛 tegrastats 로그를 짝짓던 버그 수정 | [docs/02-results.md](docs/02-results.md#6단계-성능-측정), 이 PR |
| 2026-10-26 | 8-4 물체 비교 통제 실험(컵·노트북·휴대폰·건전지·세탁기 사진, 세 모드 각 150초): 검색 1위 정답 1·2·3개로 8단계 결론을 relabel → yoloe 로 변경. 웹 화면이 열려 있으면 측정 앱이 안 꺼지던 문제 수정, 셀프 채점 87 → 90 | [docs/02-results.md 8-4](docs/02-results.md#8단계-웹캠-물체-추출과-검색), 문제 해결 기록 15, [PR #27](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/27) |
| 2026-10-26 | AI 가 쓴 7단계 평가 데이터 20쌍 사람 검수(민서, Y 20 / N 0), 셀프 채점 90 → 92 | [docs/04-eval-review.md](docs/04-eval-review.md), [PR #28](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/28) |
| 2026-10-26 | 혼자 할 수 있는 측정을 먼저: 사람 목소리 녹음으로 음성 3/3(문제 해결 4번 원인 확인), 9-3 녹화(4라벨 2,829프레임) → 9-6 GRU 학습(검증 1.000), 9-4 같은 대본으로 raw·smooth·GRU 비교(smooth 채택), 9-5 관심도 6/12, 8단계 시연 캡처. 측정 실패 4번에서 "화면 확인 + 준비됐어" 절차를 만듦 | [02-results 9단계](docs/02-results.md#9단계-사람-동작-인식), 문제 해결 기록 16~21 |
| 2026-10-27 | 10-26 측정 결과를 문서에 반영, 시연 화면 연결, 셀프 채점 92 → 99 | [docs/01-self-grade.md](docs/01-self-grade.md), [PR #29](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/29) |
| 2026-10-27 | 팀원 A 와 9단계 마지막 측정: 악수 3회(모두 검출, 카운트 12로 부풂), 처음 보는 사람 평가(smooth·GRU 모두 만세 6/6, GRU 손흔들기 1/3). 핫스팟 대역이 바뀌어 카메라 주소를 다시 찾고, 측정 중 재부팅으로 악수를 다시 잼. 셀프 채점 99 → 100 | [02-results 9단계](docs/02-results.md#9단계-사람-동작-인식), 문제 해결 기록 22·23, 이 PR |
| 2026-10-27 | 10단계 시작: 2호기 점검(VIGI·Tailscale·패키지 정상), 1호기 USB 카메라 확인(`/dev/video0`), 8091 도 사용 중이라 영상 서버 포트를 8140 으로 변경 | 문제 해결 기록 14, 이 PR |
| 2026-10-27 | 10단계 1일차: 렌즈 보정(1호기 0.063 · 2호기 1.81px), 바닥 보드 위치 보정 4번째 배치에서 성공(재현성 0.1cm). ankle1 0.52m 오차 0. 보드가 작아 위치가 2m 흔들림, 책상이 발목을 가림, 공간 0.52m 등 실패 7건 기록. 앱 메모리 기록 버그 수정 | [10단계](docs/02-results.md#10단계-두-카메라-접근-감지-진행-중-2026-10-27-1일차), 문제 해결 24~30, 이 PR |

### 회고 (2026-10-26)

- **잘된 점**: 모든 숫자를 `results/*.jsonl` 과 tegrastats 로그에서만 가져와 문서와 측정값이 어긋나지 않았습니다. 실패를 바로 기록해 문제 해결 과정이 14건 쌓였고, 그중 묶음 크기·1 epoch 결정은 숫자로 판단할 수 있었습니다.
- **아쉬운 점**: 8·9단계를 "일단 돌려 보기" 로 측정해서, 강의자료가 요구한 통제된 실험(같은 물체를 같은 순서로, 동작을 정해진 횟수만큼)이 빠졌습니다. 그래서 측정은 했는데 자료의 표는 채우지 못했습니다. 또 1~9단계를 하루에 몰아서 해 날짜별 기록이 하루뿐입니다.
- **다음에 바꿀 것**: 측정 전에 자료의 결과표를 먼저 열어 놓고 그 행·열에 맞게 시나리오를 정합니다. 재측정은 [점검 문서 3절](docs/01-self-grade.md#3-jetson-에서-다시-재야-하는-것) 순서대로 하고, 하는 날마다 이 표에 한 줄씩 남깁니다.

### 회고 (2026-10-27)

- **바꾼 것이 효과가 있었는지**: 10-26 에는 결과표를 먼저 열고 대본(만세 3·한팔들기 3·손흔들기 3·측면 만세 3)을 정한 뒤 쟀습니다. 덕분에 9-4 표를 숫자로 채웠고, "검증 1.000 인 GRU 가 실제로는 규칙보다 16번 더 틀린다" 는 비교가 가능했습니다.
- **새로 배운 것**: 원격 측정은 화면 밖 상황을 모르는 것이 가장 큰 위험이었습니다. 너무 가까운 구도, 뒤쪽 사람, 늦은 시작, 녹화 라벨 착각으로 측정 5번을 버렸습니다. 매번 미리보기로 사람 수·위치·머리 위 여백을 숫자로 확인하고, 본인이 "준비됐어" 라고 한 뒤에만 시작하자 그 뒤로는 다시 잰 일이 없었습니다.
- **남은 한계**: 9단계 학습 데이터가 한 사람·한 자리입니다. 10-27 팀원 A 평가에서 만세는 일반화됐지만 손 흔들기는 사람마다 속도가 달라 규칙은 부풀리고 GRU 는 놓쳤습니다. 여러 사람 녹화로 기준을 다시 정하는 것이 다음 단계입니다.
- **측정 운영에서 배운 것 (10-27)**: 네트워크(핫스팟 대역)가 바뀌거나 장비를 재부팅하면 그 회차 기록과 임시 폴더가 사라집니다. 시작 전 점검에 와이파이·카메라 주소를 넣고, 결과 묶음은 재부팅에도 남는 곳에 두도록 바꿨습니다.

## AI 활용과 검증

- 단계: 강의자료 요약, 장비 차이 분석, 설치 스크립트와 측정 코드 작성에 Claude Code 를 썼습니다.
- 검증: 클라우드 환경에서 Python 문법 검사, 셸 문법 검사, 가짜 측정 파일로 `results_table.py` 출력 확인까지 했습니다. 2호기 실기 실행은 원격으로 진행했고, 측정값은 장비가 남긴 `results/*.jsonl` 과 tegrastats 로그를 그대로 옮겨 커밋했습니다 (로그는 필요한 항목만 뽑은 축약본).
- AI 가 놓친 점: 처음 만든 `.gitignore` 의 `.env.*` 규칙이 `.env.example` 까지 막는 것을 리뷰 중에 발견해 예외 규칙을 넣었습니다. 처음 스크립트는 sudo 비밀번호가 필요한 상황, NVIDIA 판 ffmpeg, 1초 간격 샘플의 한계를 예상하지 못했고, 모두 실기 실행에서 드러나 고쳤습니다 (문제 해결 기록 1~3). AI 가 만든 `mask()` 는 사용자 이름에 `@` 가 있는 경우를 시험하지 않아 장비 터미널에 비밀번호를 찍었습니다 (문제 해결 기록 6). 원격 연결 상태를 잘못 읽어 "재부팅으로 끊겼다"고 잘못 안내한 일도 있었는데, `uptime -s` 와 `last -x reboot` 로 재부팅이 없었음을 확인해 바로잡았습니다.
- 9단계 측정 진행 (2026-10-26): AI 가 원격으로 미리보기 화면의 관절 좌표(어깨 y, 머리 위 여백, 사람 수)를 읽어 구도를 안내하고, 측정 뒤 이벤트 시각을 대본과 맞춰 정답 여부를 셌습니다. AI 의 실수도 있었습니다. 시작 신호가 30초 늦었다고 잘못 분석했다가 로그 시각을 다시 맞춰 정정했고, 녹화 라벨을 알리지 않아 빈 만세 녹화가 생겼습니다(문제 해결 기록 18·19). 그 뒤로 본인 확인("준비됐어") 없이는 시작하지 않게 바꿨습니다.
- 평가 데이터 사람 검수 (2026-10-26): AI 가 쓴 `data/eval.jsonl` 60쌍 중 20쌍(시드 0 무작위)을 민서가 "교육생이 할 법한 질문인가, 이 문서 하나만 정답인가"로 판정해 20쌍 모두 맞다고 봤습니다. 애매했던 1건(같은 문서를 쓰는 질문)은 기준을 다시 확인해 정리했습니다([docs/04-eval-review.md](docs/04-eval-review.md)).
- 결과 점검 (2026-10-26): 강의자료 원본의 결과표·확인 포인트·확장 과제를 AI 가 저장소의 `results/*.jsonl`·README 와 대조해 빈칸을 찾았습니다([docs/01-self-grade.md](docs/01-self-grade.md)). 검증은 표의 상태 칸을 스크립트로 다시 세는 방식으로 했고, AI 가 처음 적은 집계(충족 22 · 미실시 9)가 실제 표(충족 19 · 미실시 12)와 달라 고쳤습니다. 새로 넣은 옵션(`--video-frames`, `--mixed`, `test_all.py`, `--probe`)은 10-26 2호기 측정에서 실제로 동작을 확인했습니다.

## 역할

| 누구 | 한 일 |
| --- | --- |
| 민서 | 과제 진행 전체, 2호기·카메라 연결, 측정 대상(5단계 목소리 녹음, 8단계 물건 보여 주기, 9단계 동작·관심도 연기), 7단계 평가 데이터 사람 검수, 결과 확인 |
| 팀원 A | 10-27 9단계 악수(2명)와 처음 보는 사람 평가의 측정 대상 |
| Claude Code | 측정·설치 코드와 문서 작성, 2호기 원격 실행과 구도 확인, 결과 정리(자세한 내용은 위 "AI 활용과 검증") |

## 개인정보와 안전

- 카메라 녹화·이미지(`media/`)는 사람이 찍힐 수 있어 git 에 올리지 않습니다.
- 공용 장비 규칙(재부팅, `nvpmodel`/`jetson_clocks` 변경, `apt upgrade` 금지)은 팀 저장소 `jetson_orin` 의 CLAUDE.md 를 따릅니다. 2호기는 단독 사용 장비입니다.
