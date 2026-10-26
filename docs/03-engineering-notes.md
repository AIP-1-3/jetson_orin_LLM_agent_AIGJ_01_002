# 기술적 고민과 문제 해결 과정

실습 중 내린 판단(무엇 / 왜 / 어떻게 / 결과)과, 실패 → 원인 → 해결 → 재발 방지 기록입니다. 숫자의 출처는 [02-results.md](02-results.md) 입니다.

## 기술적 고민 (무엇 / 왜 / 어떻게 / 결과)

| 무엇 | 왜 (대안과 버린 이유) | 어떻게 | 결과 |
| --- | --- | --- | --- |
| JetPack 버전 자동 판별 | 자료의 jp6/cu126 인덱스는 Python 3.10 전용이라 JetPack 7(Python 3.12)에서 `from versions: none` 으로 실패 (자료 Q1). 수동 선택은 실수 여지가 있음 | `/etc/nv_tegra_release` 의 R 번호로 분기 (R36 → 6, R38·R39 → 7) | 2호기(R39)에서 cu132 휠이 바로 설치돼 `torch 2.14.1+cu132`, 이후 모든 측정이 GPU 에서 동작 |
| torchaudio 제외 | cu132 인덱스에 torchaudio 가 없어 PyPI cu130 판이 섞이면 transformers import 가 깨짐 (자료 Q7). 세 패키지를 모두 cu130 으로 맞추는 대안은 커널 재검증 부담이 큼 | 오디오는 soundfile·librosa 로 읽음 | 음성 검색 2/3, WAV 3개 427.9 ms 로 오디오 경로 동작 (5단계) |
| 지연은 두 번째 실행 값 | 첫 실행은 CUDA 초기화가 섞여 비교가 불공정 | 데우기 1회 후 `torch.cuda.synchronize()` 로 감싸 측정 | 텍스트 질의 2개 150.6~151.9 ms, 차원 간 차이 1.3 ms 로 안정 (6단계 표) |
| RAM 은 시스템 전체 사용량 | Jetson 은 CPU·GPU 가 메모리를 공유해 프로세스 RSS 에 GPU 사용량이 안 보임 | `/proc/meminfo` 의 MemTotal - MemAvailable + torch 할당량 + tegrastats 로그 | 데스크톱이 쓰던 약 4.7 GB 를 분리해 모델 증가분(텍스트 579 MB, 오디오 948 MB)을 따로 봄. 7단계 OOM 원인을 찾은 근거(문제 해결 기록 11) |
| 색인은 묶음 32 로 | 묶음 1 은 GPU 20% 로 처리량 6.5 질의/초. 묶음을 더 키우면 지연이 늘어 응답이 늦어짐 | 문서·크롭 색인은 묶음으로, 실시간 질의는 1건씩 (묶음 크기 실험 표) | 처리량 6.5 → 145.9 질의/초(22배), 질의당 에너지 1.47 → 0.12 J |
| 미세 조정은 1 epoch | 3 epoch 는 loss 만 더 내려가고 평가 지표는 epoch 1 과 같았음. 일반 질의 점수만 더 깎이고 학습 시간은 3배 | `EPOCHS=1` 로 학습, `check_general.py` 로 일반 성능 회귀를 함께 확인 (7단계 비교 표) | acc@1 0.850 동일, 학습 216.6 → 74.8 초, 일반 질의 점수 하락폭 절반 이하 |
| RTSP 최신 프레임만 유지 | OpenCV 버퍼에 프레임이 쌓이면 처리 지연이 수 초로 늘어남 | 읽기 전용 스레드 + 끊기면 재연결 (`lab/camera.py`) | 카메라 28 fps 수신, 검출 11.8~13.8 fps 에서도 지연 누적 없이 120초 측정 완료 (8·9단계) |
| 카메라 계정은 `.env` | 코드·git·로그에 계정이 남으면 유출 | `.env` 는 git 제외, 로그에는 `rtsp://***@` 로 가림 | 결과 파일에는 `rtsp://<카메라>/stream2` 만 남음. `@` 든 계정에서 비밀번호가 찍힌 버그를 찾아 고침(문제 해결 기록 6) |
| 동영상은 1 fps 프레임으로 대체 | `{"video"}` 입력은 torchcodec 가 필요한데 설치 목록에 없음(문제 해결 기록 9). 시스템 ffmpeg 는 NVIDIA 판이라 추출 불가 | OpenCV 로 1 fps 프레임을 뽑아 이미지로 넣고 평균 벡터를 동영상 벡터로 (`test_vision.py --video-frames`) | Jetson 실행 전 (점검 문서 3절 3번) |

## 문제 해결 과정 (실패 → 원인 → 해결 → 재발 방지)

### 1. 원격 실행에서 sudo 비밀번호 때문에 torch 설치가 멈춤 (2026-10-23, 2호기)

- **증상**: `./setup/02_install_torch.sh` 가 venv 를 만들기 전에 종료 (exit 1)
  ```
  설치 필요: ffmpeg
  libopenblas-dev (관리자 계정이 아니면 관리자에게 요청)
  sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper
  sudo: a password is required
  ```
- **원인**: 2호기에 `ffmpeg`, `libopenblas-dev` 가 없었고, 스크립트가 원격(비대화식) 세션에서 `sudo apt install` 을 불러 비밀번호를 받을 수 없었습니다. 이어서 `01_check_env.sh` 로 2호기에는 CUDA 툴킷(`/usr/local/cuda`), cuDNN, TensorRT 가 설치돼 있지 않다는 것도 확인했습니다 (1호기는 설치돼 있음).
- **해결**: 관리자 계정 터미널에서 빠진 패키지를 설치합니다. 스크립트는 비대화식이고 `sudo -n` 이 안 되면 설치 명령을 안내하고 끝나도록 바꿨습니다.
- **재발 방지**: `01_check_env.sh` 가 빠진 apt 패키지와 CUDA 툴킷 유무를 sudo 없이 먼저 알려 줍니다. 02 를 돌리기 전에 알 수 있습니다.

### 2. 경보음·잡음 샘플을 만들 수 없음: NVIDIA 판 ffmpeg 에 lavfi 없음 (2026-10-23, 2호기)

- **증상**: `./scripts/make_audio_samples.sh` 가 경보음 단계에서 종료되어 `test_audio.py` 가 "wav 가 없습니다"로 끝남
  ```
  [in#0 @ 0xaaaae5cbefc0] Unknown input format: 'lavfi'
  Error opening input file sine=frequency=1000:duration=3.
  Error opening input files: Invalid argument
  ```
- **원인**: `nvidia-jetpack` 설치 때 함께 들어온 NVIDIA 판 ffmpeg(7:8.0.1-nvidia1)는 꼭 필요한 기능만 넣어 빌드돼서 lavfi 입력과 sine·anoisesrc 필터가 없습니다. 강의자료의 명령은 Ubuntu 판 ffmpeg 를 가정합니다.
- **해결**: 경보음(1 kHz 3초)과 핑크 잡음(5초)을 numpy 로 만들고, 16 kHz 모노 변환도 librosa 로 합니다 (`lab/make_audio_samples.py`). 시스템 ffmpeg 를 Ubuntu 판으로 되돌리는 대안은 Jetson 영상 가속 패키지와 얽혀 있어 버렸습니다.
- **재발 방지**: 테스트 자료 생성은 시스템 도구 대신 venv 안의 파이썬 패키지로 해서 장비마다 ffmpeg 빌드가 달라도 같은 파일이 나옵니다 (잡음은 시드 0 고정).

### 3. 1초 간격 tegrastats 에 GPU 사용률이 0% 로 찍힘 (2026-10-23, 2호기)

- **증상**: 텍스트 테스트 중 tegrastats 24줄 가운데 23줄이 `GR3D_FREQ 0%`, 1줄만 12%
- **원인**: 질의 2개 추론이 약 150 ms 라서 1초마다 한 번 찍는 샘플에 거의 걸리지 않습니다. GPU 를 안 쓴 것이 아니라 측정 방법의 한계입니다.
- **해결**: `test_text.py --sustain 10` 으로 10초 동안 연속 추론하며 처리량(질의/초)을 재고, `TS_MS=200` 으로 tegrastats 간격을 줄입니다.

### 4. 합성 음성을 "냉방 요청 목소리"로 찾지 못함 (2026-10-23, 2호기, 미해결)

- **증상**: `test_audio.py` 에서 3개 중 1개 틀림
  ```
  [NG] '덥다고 냉방을 요청하는 목소리' -> noise.wav (0.651)
  [OK] '날카로운 경보음' -> beep.wav (0.702)
  [OK] '웅웅거리는 배경 소음' -> noise.wav (0.682)
  ```
- **원인 (추정)**: espeak-ng 합성음은 사람 목소리보다 기계음에 가까워서 "목소리" 질의와 덜 닮았을 수 있습니다. 강의자료도 합성음은 점수가 낮게 나올 수 있다고 적고 있습니다. 아직 확인하지 않았습니다.
- **확인 계획**: 사람 목소리를 녹음해 같은 질의로 비교하고, 질의 문구("사람이 말하는 소리")를 바꿔 결과가 바뀌는지 봅니다.

### 5. IP 카메라 녹화를 ffmpeg 로 할 수 없음: NVIDIA 판 ffmpeg 에 RTSP·libx264 없음 (2026-10-23, 2호기)

- **증상**: 강의자료 방식(`ffmpeg -rtsp_transport tcp -i rtsp://... -c:v libx264`)으로 쓴 첫 `record_clip.sh` 를 실행하기 전 점검에서, 2호기 ffmpeg 의 `-protocols` 목록에 `file`, `pipe` 만 있고 rtsp·tcp 가 없으며 인코더 목록에 libx264 가 없음을 확인
- **원인**: 문제 해결 기록 2 와 같은 NVIDIA 판 ffmpeg(7:8.0.1-nvidia1)가 네트워크 기능을 끄고(`--disable-network`) 빌드돼 있습니다.
- **해결**: OpenCV 는 시스템 ffmpeg 가 아닌 자체 FFmpeg 라이브러리로 RTSP 를 열 수 있으므로, `lab/camera.py` 의 `open_capture()` 로 받아 `cv2.VideoWriter`(mp4v)로 저장합니다 (`lab/record_clip.py`). 화면 출력에는 `mask()` 로 계정을 가립니다.
- **재발 방지**: 녹화·변환처럼 장비 ffmpeg 빌드에 따라 달라지는 일은 venv 안의 OpenCV·PyAV 로 처리합니다. 실제 카메라 녹화 결과(받은 fps, 해상도)는 연결 시험 후 이 항목에 붙입니다.

### 6. 계정 가리기(`mask()`)가 비밀번호를 가리지 못함 (2026-10-23, 2호기)

- **증상**: 첫 카메라 연결 시험에서 `python camera.py` 가 찍은 "소스:" 줄에 카메라 비밀번호가 보였습니다. 장비의 작업 터미널에만 찍혔고 채팅·저장소에는 들어가지 않았습니다.
- **원인**: 카메라 사용자 이름이 이메일 형식이라 `@` 가 들어 있었습니다. `mask()` 가 첫 `@` 로 계정과 주소를 나눠서, 사용자 이름 앞부분만 가리고 나머지(사용자 이름 뒷부분과 비밀번호)는 주소로 보고 그대로 찍었습니다.
- **해결**: 마지막 `@` 로 나누도록 바꿨습니다 (`rsplit`). `.env.example` 에는 계정 속 `@` 를 `%40` 으로 적으라는 안내를 넣었습니다. URL 규칙상 그래야 RTSP 주소도 바르게 해석됩니다. 노출된 비밀번호는 바꾸도록 권했습니다.
- **검증**: `@` 가 든 사용자 이름, `%40` 으로 적은 이름, 일반 이름의 세 주소로 `mask()` 결과에 계정이 남지 않음을 확인했습니다.
- **재발 방지**: 계정이 든 주소는 `mask()` 를 거친 문자열만 출력합니다. 예시 계정에 `@` 를 넣어 시험하는 습관을 둡니다.

### 7. 2호기에서 IP 카메라에 닿지 않음 (2026-10-23, 2호기, 우회 해결)

- **증상**: RTSP(554)와 웹(80) 포트 모두 TCP 연결이 시간 초과되고, ping 도 100% 손실이었습니다. 인증 단계까지 가지 못했으므로 계정 문제는 아닙니다.
- **원인 (추정)**: 2호기는 10.220.x.x 대역이고 카메라는 10.108.x.x 고정 IP 입니다. 2호기는 카메라로 가는 패킷을 기본 게이트웨이로 보내는데, 그 경로가 막혀 있습니다. 앞서 ONVIF 장치 찾기(TTL 1 멀티캐스트)에는 카메라가 응답했습니다. 그래서 카메라는 같은 스위치에 있고 IP 만 다른 대역으로 설정된 것으로 봅니다.
- **원인 확인**: 2호기에 카메라 대역 보조 IP 를 잠시 붙여 직접 경로를 만들어도 ARP 응답이 없었습니다. 카메라는 학교가 자리 랜선 포트에 배정한 10.108.x.x 고정 IP 로 설정돼 있었고, 그 포트는 2호기와 다른 VLAN 이라 학교 라우터를 거쳐야 하는데 2호기 → 카메라 방향이 막혀 있었습니다. 보조 IP 는 확인 후 바로 지웠습니다.
- **시도했다 실패한 것**: 카메라를 학교 와이파이로 옮겼지만, 로그인 페이지(캡티브 포털)를 카메라가 통과할 수 없어 클라우드 등록이 안 됐고, 같은 와이파이의 휴대폰에서도 찾지 못했습니다(기기 간 통신 차단으로 추정).
- **해결(우회)**: 카메라를 초기화하고 휴대폰 핫스팟으로 VIGI 앱에 등록한 뒤, 2호기 와이파이도 같은 핫스팟에 붙였습니다. ping 3/3(평균 18 ms), RTSP 554 포트 열림을 확인했습니다. 계정은 앱 로그인 계정이 아니라 카메라 장치 계정(`admin` + 장치 비밀번호)을 써야 했습니다(앱 계정으로는 401).
- **남은 위험**: 핫스팟이 꺼지면 카메라가 끊깁니다. 2호기의 원격 접속은 유선으로 유지하고 와이파이는 기본 경로로 쓰지 않게(`ipv4.never-default`) 해야 합니다. 실제로 유선이 빠진 상태에서 핫스팟 설정을 바꾸다 원격 접속이 한 번 끊겼습니다. 장기적으로는 학교에 2호기 → 카메라 RTSP(554) 허용을 요청하는 것이 맞습니다.

### 8. 실습 venv 에 OpenCV 가 없음 (2026-10-23, 2호기)

- **증상**: `python camera.py` 가 `ModuleNotFoundError: No module named 'cv2'` 로 실패
- **원인**: venv 를 `--system-site-packages` 없이 만들어 JetPack 의 시스템 OpenCV 를 보지 못합니다. 설치 스크립트 목록에도 OpenCV 가 없었습니다.
- **해결**: torch 고정 파일을 지키며(`-c torch-pin.txt`) `opencv-python-headless` 를 설치했고, `setup/03_install_packages.sh` 에도 넣었습니다. 이 휠은 자체 FFmpeg 를 갖고 있어 NVIDIA 판 ffmpeg 와 관계없이 RTSP 를 엽니다. 설치 후에도 torch 2.14.1+cu132 가 그대로임을 확인했습니다.

### 9. 동영상 임베딩 실패: torchcodec 없음 (2026-10-23, 2호기, 미해결)

- **증상**: `test_vision.py` 가 이미지 5장 검색 뒤 `clip.mp4` 에서 멈춤
  ```
  ImportError: `torchcodec` is not installed and cannot be used to decode the video by default. Video decoding with `torchvision` is not available: `torchvision.io.read_video` was deprecated in `torchvision==0.22` and removed in `torchvision==0.26.0`.
  ```
- **원인**: sentence-transformers 의 동영상 입력은 torchcodec 로 디코딩하는데, 설치 목록에 없었습니다. 예전 대안이던 torchvision 의 `read_video` 는 제거됐습니다.
- **임시 조치**: `--no-video` 로 이미지 검색만 측정했습니다. 실패한 실행의 tegrastats 로그는 표에 섞이지 않게 장비의 백업 폴더로 옮겼습니다.
- **다음 단계**: torchcodec 는 FFmpeg 공유 라이브러리가 필요한데, 2호기의 시스템 ffmpeg 는 NVIDIA 판이라 맞는 라이브러리를 깔려면 관리자 계정의 sudo apt 가 필요합니다. 그래서 sudo 없이 되는 길을 먼저 만들었습니다. `test_vision.py --video-frames` 가 OpenCV(pip 휠에 FFmpeg 포함)로 1 fps 프레임을 뽑아 이미지로 넣고, 평균 벡터를 동영상 벡터로 씁니다. 모델의 동영상 입력도 기본 1 fps 샘플링이라 같은 정보를 보지만, 프레임 사이 시간 관계를 모델이 직접 보지는 못한다는 차이가 있습니다. 2호기 실행 결과는 아직 없습니다.

### 10. 이미지 10장을 한 번에 넣자 RAM 97% (2026-10-23, 2호기)

- **증상**: 20초 녹화에서 뽑은 10장을 `model.encode()` 한 번에 넣었더니 tegrastats RAM 이 최대 7262/7485 MB 까지 올랐습니다 ([`results/logs/04_vision_20261023-155610.log`](../results/logs/04_vision_20261023-155610.log)). torch 최대 할당은 1593 MB 로 5장(1225 MB)보다 368 MB 늘었습니다.
- **원인**: 통합 메모리라 이미지 인코더의 중간 텐서가 시스템 RAM 을 그대로 씁니다. 데스크톱 세션이 이미 약 4.7 GB 를 쓰고 있어 여유가 적었습니다.
- **해결**: `test_vision.py` 에 `IMAGE_BATCH = 4` 를 두고 `batch_size=IMAGE_BATCH` 로 나눠 넣습니다([PR #11](https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002/pull/11)). 측정 기록에도 묶음 크기를 남깁니다. 4장씩 나눠 넣은 뒤 6장 측정의 최대 RAM 은 6489 MB(87%), torch 최대 할당 1156 MB 였습니다([`04_vision_20261023-160530.log`](../results/logs/04_vision_20261023-160530.log)). 장수가 달라 직접 비교는 아니지만, 장당 시간은 약 0.5~0.57 초로 그대로입니다.

### 11. 7단계 미세 조정이 두 번째 step 에서 메모리 부족 (2026-10-23, 2호기, 해결)

- **증상**: `train_ft.py` 를 자료 기본값(배치 8×누적 2)으로 돌리자 첫 step(1/54, 7.6초)은 끝났지만 두 번째 step 의 `loss.backward` 에서 실패했습니다. 오류는 `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB ... 152.73 MiB is free` 이고, 그 앞에 `NvMapMemAllocInternalTagged failed: error 12` 가 찍혔습니다. 자료가 권하는 배치 4×누적 4 로 바꿔도 같은 지점(1/54 뒤)에서 같은 오류가 났습니다. 두 번 모두 tegrastats RAM 이 7195~7237/7485 MB(96~97%)까지 찼습니다([`07_finetune_20261023-161440.log`](../results/logs/07_finetune_20261023-161440.log)).
- **원인**:
  - 학습 전부터 약 4.3 GB 가 쓰이고 있었습니다. Firefox 약 0.9 GB, gnome-shell 0.33 GB, 시스템 모니터 0.23 GB, Xorg 0.21 GB 등입니다. 그래서 학습에 남은 메모리가 약 3 GB 였습니다.
  - 모델 파라미터 271.0M 중 134.2M(49.5%)이 토큰 임베딩 표입니다. 이 표까지 학습하면 그만큼 기울기와 옵티마이저 상태가 더 생깁니다(파라미터 수는 재학습 로그의 출력).
- **해결 시도**: 배치를 더 줄이면 MultipleNegativesRankingLoss 의 오답 수도 줄어 학습 효과가 작아집니다(자료 확인 포인트). 그래서 배치는 그대로 두고 메모리가 큰 쪽을 줄였습니다.
  - `FREEZE_EMB=1`: 토큰 임베딩 표를 고정합니다. 도메인 문장의 단어는 이미 어휘에 있어 표를 고칠 필요가 작습니다.
  - `MAX_LEN=128`: 학습 문장의 최대 길이를 줄입니다.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: 메모리 단편화를 줄입니다. 오류 메시지가 권하는 설정입니다.
- **결과**: 세 가지를 적용하자 자료 기본 배치(8×누적 2) 그대로 54 step 을 끝까지 마쳤습니다(216.6 초). 다만 RAM 최대 7156/7485 MB(96%)에 스왑 약 1.2 GB 를 써 여유는 거의 없었습니다([`07_finetune_20261023-161852.log`](../results/logs/07_finetune_20261023-161852.log)). 정확도 결과는 7단계 절에 있습니다.
- **배운 점**: 8 GB 통합 메모리에서는 데스크톱 세션이 쓰는 몫까지 계산해야 합니다. 학습을 자주 돌린다면 데스크톱을 끄는 헤드리스 운영이나 PC·Colab 에서 학습하고 Jetson 은 추론만 하는 방식(자료 권장)이 맞습니다.

### 12. 8단계 앱이 기록 뒤 비정상 종료 (2026-10-23, 2호기, 해결 확인)

- **증상**: `app.py --seconds 60` 이 결과를 `08_objects.jsonl` 에 쓴 직후 `terminate called without an active exception` 을 찍고 exit 134 로 끝났습니다. yolo·relabel 두 번 모두 같았습니다. Traceback 은 없습니다.
- **원인**: 검출·임베딩 데몬 스레드가 CUDA·OpenCV 작업을 하는 도중에 파이썬 인터프리터가 정리되면서 C++ 쪽에서 abort 가 났습니다. 측정값은 이미 저장된 뒤라 숫자에는 영향이 없지만, `gpu-run.sh` 나 스크립트가 실패로 보게 됩니다.
- **해결**: 기록(`finish()`)과 카메라 정리(`cam.close()`) 뒤 `os._exit(0)` 으로 바로 끝내도록 `app.py`·`pose_app.py` 를 고쳤습니다. 같은 구조인 9단계에도 미리 적용했습니다.
- **함께 고친 것**: 첫 실행 때 ultralytics 가 추적용 `lap`, yoloe 용 `clip`·`ftfy` 를 실행 중에 자동 설치했습니다. 실행 도중 네트워크 설치가 일어나지 않도록 `setup/04_install_detect.sh` 에 넣었습니다. yoloe 가 작업 폴더에 받는 `mobileclip_blt.ts`(600 MB)는 `.gitignore` 에 추가했습니다.
- **확인**: 수정 뒤 yoloe 1회, 9단계 2회가 모두 exit 0 으로 끝났습니다.

### 13. 9단계 손 흔들기가 한팔들기와 번갈아 찍힘 (2026-10-23, 2호기, 해결)

- **증상**: 기본 모드에서 손을 흔드는 동안 이벤트가 "한팔들기 → 손흔들기 → 한팔들기" 로 1초 안에 번갈아 기록됐습니다. 이벤트 41개 중 18쌍이 그랬고, t=112.3·112.8·119.5 초처럼 같은 0.1초에 둘 다 찍힌 곳도 있습니다([`results/09_pose.jsonl`](../results/09_pose.jsonl)).
- **원인**: 손 흔들기는 최근 1.5초 동안 손목이 좌우로 2번 이상 방향을 바꿨는지를 매 프레임 다시 계산합니다. 손이 방향을 바꾸는 사이에는 조건이 잠깐 깨져 다시 한팔들기가 되고, 라벨이 바뀔 때마다 이벤트를 세기 때문에 횟수가 부풀었습니다.
- **해결**: 손 흔들기를 한 번 잡으면 `WAVE_HOLD_S = 1.0` 초 동안 유지하도록 `lab/pose_app.py` 에 히스테리시스를 넣었습니다.
- **결과**: 같은 동작을 2분 다시 쟀습니다(앉은 자세라 조건이 완전히 같지는 않음). 이벤트 순서에서 세어 보면 다음과 같습니다.

| 측정 | 이벤트 수 | 1초 안에 한팔들기↔손흔들기가 바뀐 쌍 | 그중 손흔들기 → 한팔들기로 되돌아간 쌍 | 같은 0.1초에 두 개 |
| --- | --- | --- | --- | --- |
| 수정 전 (6540d00) | 41 | 18 | 5 | 4 |
| 수정 후 (05c80af) | 37 | 9 | 1 | 0 |

  한팔들기 → 손흔들기 순서의 쌍은 팔을 든 뒤 흔들기 시작하는 정상 전환이라 남는 것이 맞습니다. 깜빡임에 해당하는 되돌아감은 5 → 1 로 줄었고, 남은 1개(37.1 → 38.1 초)는 정확히 유지 시간 1초 경계에서 생겼습니다.

### 14. 10단계에서 2호기가 1호기를 찾지 못함 (2026-10-23, 1·2호기, 해결)

- **증상**: 2호기의 `tailscale status` 에 1호기(aip-1-3)가 없고 `tailscale ping <1호기 Tailscale 주소>` 결과가 `no matching peer` 였습니다. 1호기 목록에도 2호기가 없었습니다. 맥북에서는 두 장비 모두 접속됐습니다.
- **원인**: 두 장비는 같은 tailnet 에 있었지만, 팀 접근 규칙이 "팀원 → tag:jetson" 만 허용하고 "tag:jetson → tag:jetson" 규칙이 없었습니다. Tailscale 은 허용 규칙이 없는 기기끼리는 서로 목록에서 숨깁니다. 처음에는 계정이 다르다고 보고 기기 공유를 시도했지만, 같은 tailnet 이라 초대 수락이 실패했습니다. 계정 확인에 쓴 `grep -m1 LoginName` 은 tailnet 의 첫 번째 사용자를 보여 줄 뿐이라 판단을 잘못 이끌었고, `CurrentTailnet.Name` 으로 다시 확인했습니다.
- **해결**: 관리 화면에서 `tag:jetson → tag:jetson` 규칙을 추가했습니다. 바로 2호기에 1호기가 보였고 `tailscale ping` 은 DERP(도쿄) 중계로 92~103 ms 였습니다. 2호기는 휴대폰 핫스팟, 1호기는 학교망이라 직접 연결은 맺어지지 않았습니다.
- **함께 고친 것**: 1호기 8090 포트는 다른 프로젝트의 센서 수집 서버가 이미 쓰고 있었습니다. 10단계 영상 서버(`prox/cam_server.py`)는 8091 로 옮기고, 접근 규칙도 그 포트만 열도록 바꿨습니다.
