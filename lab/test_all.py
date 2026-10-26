"""5단계 확장 과제 + 6단계 '전체 (740M) / 혼합' 행: 이미지·동영상 프레임·음성을 한 인덱스에 넣고 한 질의로 찾는다.

전체 모델(텍스트+비전+오디오)을 올리므로 8GB 공유 메모리에서 가장 무겁다. 실행 전에 free -h 로 여유를 보고,
데스크톱 브라우저 같은 큰 프로그램은 닫는다. 이미지는 test_vision.py 와 같이 4장씩 나눠 넣는다.
정답 판정: 음성 질의는 파일 이름, 이미지 질의는 장면별 녹화 파일 이름 앞부분(empty_/people_/board_)으로 자동 판정한다.
"""
import glob
import tempfile

import soundfile as sf

from bench import Recorder, media_dir
from eg2 import config_name, load
from test_vision import IMAGE_BATCH, video_frames

# (질의, 정답 판정: 파일 이름이 이 값으로 시작하면 정답)
QUERIES = [("덥다고 냉방을 요청하는 목소리", "speech.wav"),
           ("날카로운 경보음", "beep.wav"),
           ("웅웅거리는 배경 소음", "noise.wav"),
           ("사람이 없는 빈 강의실", "empty_"),
           ("학생들이 앉아 수업을 듣는 모습", "people_"),
           ("화이트보드에 쓴 글씨", "board_")]


def main():
    media = media_dir()
    images = sorted(glob.glob(str(media / "*.jpg")))
    wavs = sorted(str(p) for p in media.glob("*.wav") if not p.name.startswith("raw_"))
    if not images or not wavs:
        raise SystemExit(f"{media} 에 jpg 와 wav 가 모두 있어야 합니다 (4·5단계 자료를 먼저 만드세요)")
    for f in wavs:
        info = sf.info(f)
        assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님"

    rec = Recorder("06_all", config_name(vision=True, audio=True))
    model = load(vision=True, audio=True)
    rec.loaded()

    names = [p.split("/")[-1] for p in images] + [p.split("/")[-1] for p in wavs]
    inputs = [{"image": p} for p in images] + [{"audio": p} for p in wavs]
    with tempfile.TemporaryDirectory() as tmp:
        clip = media / "clip.mp4"
        frames = video_frames(clip, tmp)[0] if clip.exists() else []
        # 동영상은 1 fps 프레임으로 넣는다 (torchcodec 없음, README 문제 해결 기록 9)
        names += [f"clip.mp4@{i}s" for i in range(len(frames))]
        inputs += [{"image": p} for p in frames]

        model.encode(inputs[:1], normalize_embeddings=True)   # 데우기
        with rec.time(f"혼합 {len(inputs)}개 (이미지 {len(images)}, WAV {len(wavs)}, 프레임 {len(frames)})",
                      n=len(inputs), batch=IMAGE_BATCH):
            emb = model.encode(inputs, batch_size=IMAGE_BATCH, normalize_embeddings=True)

    with rec.time(f"질의 {len(QUERIES)}개"):
        q = model.encode([qi for qi, _ in QUERIES], prompt_name="SearchQuery", normalize_embeddings=True)
    for (qi, answer), row in zip(QUERIES, q @ emb.T):
        order = row.argsort()[::-1]
        top = [(names[j], round(float(row[j]), 3)) for j in order[:3]]
        if not any(n.startswith(answer) for n in names):
            print(f"  [건너뜀] '{qi}': 정답 파일({answer}*)이 인덱스에 없음 {top}")
            continue
        rec.check(f"'{qi}'", names[order[0]].startswith(answer), str(top))
    rec.save()


if __name__ == "__main__":
    main()
