# 과제 제출 저장소로 옮기기

강사님 안내(2026-11-02)에 따라 2주간 진행한 과제 3개를 AIP-1 조직의 `<과제이름>-AIGJ_01_002` 저장소에 올리기 위한 폴더입니다.

| 제출 저장소 | 내용 | 원본 |
| --- | --- | --- |
| `AIP-1/timeseries-inference-AIGJ_01_002` | 3.2 시계열 추론 과제 | `AIP-1-3/jenson_AIGJ_01_002` main 그대로 |
| `AIP-1/embedding_gemma2-AIGJ_01_002` | EmbeddingGemma 2 실습 1~7단계 | 이 저장소 + `README-embedding_gemma2.md` 를 README 로 |
| `AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_002` | 같은 실습의 카메라 응용 8~10단계 | 이 저장소 + `README-camera_applications.md` 를 README 로 |

맥 터미널에서 (GitHub 에 push 할 수 있는 계정으로):

```bash
git clone https://github.com/AIP-1-3/jetson_orin_LLM_agent_AIGJ_01_002.git /tmp/eg2-submit && bash /tmp/eg2-submit/submission/push_to_aip1.sh
```

커밋 이력은 그대로 옮겨지고, 나눠 내는 두 저장소에만 README 를 바꾸는 커밋이 하나씩 더 얹힙니다. 대상 저장소에 이미 커밋이 있으면 push 가 거부되어 덮어쓰지 않습니다.
