# Jetson Orin LLM 에이전트 실습: EmbeddingGemma 2 멀티모달 임베딩

Jetson Orin Nano Super 8GB 에서 EmbeddingGemma 2 로 텍스트·이미지·음성을 한 벡터 공간에서 검색하고, IP 카메라 물체 검색·동작 인식으로 넓힌 실습 저장소입니다.

- **해결하는 문제**: 강의자료는 JetPack 6·USB 웹캠 기준이라 우리 장비(JetPack 7, IP 카메라)에서 그대로 돌지 않습니다. 장비 차이를 자동으로 판별해 설치하고, 모든 측정값을 측정 조건과 함께 파일로 남깁니다.
- **핵심 결과** (2호기, MAXN_SUPER, bfloat16): 텍스트 검색 6/6 정답·질의 2개 151 ms, 묶음 32 로 처리량 22배(6.5 → 145.9 질의/초), 도메인 미세 조정 accuracy@1 0.833 → 0.850, 물체 검출 10.9~12.9 fps, COCO 밖 물체(세탁기 사진) 검색 1위 정답 yolo 1 → relabel 2 → yoloe 3개(5개 중), 자세 인식 13.6~13.8 fps.
- **시연 링크**: 준비 중
- **문서**: [단계별 결과](docs/02-results.md) · [기술적 고민과 문제 해결 과정](docs/03-engineering-notes.md) · [강의자료 대비 점검과 셀프 채점](docs/01-self-grade.md) · [강의자료 정리](docs/00-lecture-summary.md)

## 단계별 결과 한눈에

| 단계 | 결과 | 근거 |
| --- | --- | --- |
| 1~2 설치 | torch 2.14.1+cu132 (JetPack 7 자동 판별), sentence-transformers 6.2.0 | `setup/`, [문제 해결 1·8](docs/03-engineering-notes.md) |
| 3 텍스트 | 6/6 정답, 150.6~151.9 ms (768/256/128 차원), 초당 13.3 질의. 프롬프트를 빼면 3/6 | [`results/03_text.jsonl`](results/03_text.jsonl) |
| 4 이미지·동영상 | 장면별 6장 1/3 정답, 6장 2.7~2.9 초, 동영상 21초(1 fps 프레임) 9.1 초, 텍스트+이미지 혼합 548 ms | [`results/04_vision.jsonl`](results/04_vision.jsonl) |
| 5 음성 | 2/3 정답 (합성 음성 오답), WAV 3개 427.9 ms | [`results/05_audio.jsonl`](results/05_audio.jsonl) |
| 6 성능 | 모델 증가분 텍스트 579 MB · 오디오 948 MB · 전체 1270 MB(한 인덱스 검색 3/6), 묶음 32 로 처리량 22배·질의당 에너지 1/12 | [6단계](docs/02-results.md#6단계-성능-측정) |
| 7 미세 조정 | accuracy@1 0.833 → 0.850, 1 epoch 74.8 초, 일반 질의 24/24 → 21/24 | [`results/07_finetune.jsonl`](results/07_finetune.jsonl) |
| 8 물체 검색 | 같은 물체 5종 통제 실험(각 150초): 검색 1위 정답 yolo 1 · relabel 2 · yoloe 3개, fps 12.9 · 11.5 · 10.9 → 강의실용으로 yoloe 선택. relabel 은 tv → monitor 교정, 대신 맞는 cup 도 물병으로 바꿈 | [`results/08_objects.jsonl`](results/08_objects.jsonl), [8-4 표](docs/02-results.md#8단계-웹캠-물체-추출과-검색) |
| 9 동작 인식 | 13.6~13.8 fps, 손 흔들기 시간 패턴 판정, 깜빡임 되돌아감 5 → 1 | [`results/09_pose.jsonl`](results/09_pose.jsonl) |

강의자료 결과표 중 아직 비어 있는 칸(8-4 COCO 밖 물체, 9-4~9-6)과 채우는 순서는 [docs/01-self-grade.md](docs/01-self-grade.md) 에 있습니다.

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

브라우저에서 `http://<2호기 Tailscale 주소>:8000` 으로 실시간 영상과 검색 화면을 봅니다. 사람은 크롭·저장하지 않습니다(8단계 `SKIP_CLASSES`). 9단계는 관절 좌표만 다룹니다.

8·9단계 결과(강의자료 8-4·9-4 표 포함)는 [docs/02-results.md](docs/02-results.md#8단계-웹캠-물체-추출과-검색) 에 있습니다.

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
| 2026-10-26 | 8-4 물체 비교 통제 실험(컵·노트북·휴대폰·건전지·세탁기 사진, 세 모드 각 150초): 검색 1위 정답 1·2·3개로 8단계 결론을 relabel → yoloe 로 변경. 웹 화면이 열려 있으면 측정 앱이 안 꺼지던 문제 수정, 셀프 채점 87 → 90 | [docs/02-results.md 8-4](docs/02-results.md#8단계-웹캠-물체-추출과-검색), 문제 해결 기록 15, 이 PR |

### 회고 (2026-10-26)

- **잘된 점**: 모든 숫자를 `results/*.jsonl` 과 tegrastats 로그에서만 가져와 문서와 측정값이 어긋나지 않았습니다. 실패를 바로 기록해 문제 해결 과정이 14건 쌓였고, 그중 묶음 크기·1 epoch 결정은 숫자로 판단할 수 있었습니다.
- **아쉬운 점**: 8·9단계를 "일단 돌려 보기" 로 측정해서, 강의자료가 요구한 통제된 실험(같은 물체를 같은 순서로, 동작을 정해진 횟수만큼)이 빠졌습니다. 그래서 측정은 했는데 자료의 표는 채우지 못했습니다. 또 1~9단계를 하루에 몰아서 해 날짜별 기록이 하루뿐입니다.
- **다음에 바꿀 것**: 측정 전에 자료의 결과표를 먼저 열어 놓고 그 행·열에 맞게 시나리오를 정합니다. 재측정은 [점검 문서 3절](docs/01-self-grade.md#3-jetson-에서-다시-재야-하는-것) 순서대로 하고, 하는 날마다 이 표에 한 줄씩 남깁니다.

## AI 활용과 검증

- 단계: 강의자료 요약, 장비 차이 분석, 설치 스크립트와 측정 코드 작성에 Claude Code 를 썼습니다.
- 검증: 클라우드 환경에서 Python 문법 검사, 셸 문법 검사, 가짜 측정 파일로 `results_table.py` 출력 확인까지 했습니다. 2호기 실기 실행은 원격으로 진행했고, 측정값은 장비가 남긴 `results/*.jsonl` 과 tegrastats 로그를 그대로 옮겨 커밋했습니다 (로그는 필요한 항목만 뽑은 축약본).
- AI 가 놓친 점: 처음 만든 `.gitignore` 의 `.env.*` 규칙이 `.env.example` 까지 막는 것을 리뷰 중에 발견해 예외 규칙을 넣었습니다. 처음 스크립트는 sudo 비밀번호가 필요한 상황, NVIDIA 판 ffmpeg, 1초 간격 샘플의 한계를 예상하지 못했고, 모두 실기 실행에서 드러나 고쳤습니다 (문제 해결 기록 1~3). AI 가 만든 `mask()` 는 사용자 이름에 `@` 가 있는 경우를 시험하지 않아 장비 터미널에 비밀번호를 찍었습니다 (문제 해결 기록 6). 원격 연결 상태를 잘못 읽어 "재부팅으로 끊겼다"고 잘못 안내한 일도 있었는데, `uptime -s` 와 `last -x reboot` 로 재부팅이 없었음을 확인해 바로잡았습니다.
- 결과 점검 (2026-10-26): 강의자료 원본의 결과표·확인 포인트·확장 과제를 AI 가 저장소의 `results/*.jsonl`·README 와 대조해 빈칸을 찾았습니다([docs/01-self-grade.md](docs/01-self-grade.md)). 검증은 표의 상태 칸을 스크립트로 다시 세는 방식으로 했고, AI 가 처음 적은 집계(충족 22 · 미실시 9)가 실제 표(충족 19 · 미실시 12)와 달라 고쳤습니다. 새로 넣은 옵션(`--video-frames`, `--mixed`, `test_all.py`, `--probe`)은 문법 검사만 했고, 장비 실행은 아직입니다.

## 개인정보와 안전

- 카메라 녹화·이미지(`media/`)는 사람이 찍힐 수 있어 git 에 올리지 않습니다.
- 공용 장비 규칙(재부팅, `nvpmodel`/`jetson_clocks` 변경, `apt upgrade` 금지)은 팀 저장소 `jetson_orin` 의 CLAUDE.md 를 따릅니다. 2호기는 단독 사용 장비입니다.
