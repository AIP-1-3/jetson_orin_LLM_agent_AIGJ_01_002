"""4단계 영상 테스트: 텍스트 질의로 이미지·동영상 찾기.

자료(media/)는 scripts/record_clip.sh(OpenCV 녹화)로 IP 카메라에서 만들거나 직접 복사한다.
정답은 두 가지로 남긴다.
- 장면별 녹화(record_clip.sh 4 empty|people|board)로 만든 파일은 이름 앞부분이 그 장면이라,
  질의마다 1위 이미지 이름이 기대 장면으로 시작하는지 자동 판정한다. 장면 이름은 찍을 때 사람이 정했다.
- 그 밖의 파일은 사람이 상위 3개를 눈으로 보고 --judge 로 남긴다.
"""
import argparse
import glob
import tempfile

import numpy as np

from bench import Recorder, media_dir
from eg2 import config_name, load

QUERIES = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]
# 질의별 기대 장면 = 장면별 녹화 파일 이름 앞부분
EXPECTED = ["empty", "people", "board"]
# 이미지 10장을 한 묶음으로 넣으면 2호기 RAM 이 97%(7262/7485 MB)까지 찼다. 4장씩 나눠 넣는다.
IMAGE_BATCH = 4
# 확장 과제: 텍스트와 이미지를 섞은 입력 하나 (자료 4단계)
MIXED_TEXT = "강의실 <|image|>"


def video_frames(clip, out_dir, fps=1.0):
    """동영상을 초당 fps 장 jpg 로 뽑는다. 모델의 기본 동영상 샘플링(1 fps)과 맞춘다.

    {"video": ...} 입력은 torchcodec 가 있어야 디코딩되는데 aarch64·cu132 에서는 설치가 막혀 있다
    (README 문제 해결 기록 9). 시스템 ffmpeg 도 NVIDIA 판이라 쓸 수 없어 OpenCV 로 뽑는다.
    """
    import cv2
    cap = cv2.VideoCapture(str(clip))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    paths, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            path = f"{out_dir}/v_{len(paths):03d}.jpg"
            cv2.imwrite(path, frame)
            paths.append(path)
        i += 1
    cap.release()
    return paths, i / src_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="동영상 디코딩이 실패할 때 이미지만 검색")
    ap.add_argument("--video-frames", action="store_true",
                    help="동영상을 1 fps 프레임으로 뽑아 이미지로 넣고 평균 벡터를 동영상 벡터로 쓴다")
    ap.add_argument("--mixed", action="store_true", help="확장 과제: 텍스트+이미지 혼합 입력도 잰다")
    ap.add_argument("--judge", nargs="*", default=[], metavar="Y|N",
                    help="질의별 1위 이미지가 맞았는지 사람 판정 (예: --judge Y Y N)")
    args = ap.parse_args()

    media = media_dir()
    images = sorted(glob.glob(str(media / "*.jpg")))
    if not images:
        raise SystemExit(f"{media} 에 jpg 가 없습니다. scripts/record_clip.sh 로 만들어 주세요")
    clip = media / "clip.mp4"

    rec = Recorder("04_vision", config_name(vision=True))
    model = load(vision=True)
    rec.loaded()

    inputs = [{"image": p} for p in images]
    model.encode(inputs[:1], normalize_embeddings=True)   # 데우기
    with rec.time(f"이미지 {len(images)}장", n=len(images), batch=IMAGE_BATCH):
        img_emb = model.encode(inputs, batch_size=IMAGE_BATCH, normalize_embeddings=True)

    q = model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
    sim = q @ img_emb.T
    names = [p.split("/")[-1] for p in images]
    # 기대 장면이 모두 찍혀 있을 때만 자동 판정한다 (일부만 있으면 정답이 없는 질의가 생긴다)
    auto = not args.judge and all(any(n.startswith(e + "_") for n in names) for e in EXPECTED)
    for i, (qi, row) in enumerate(zip(QUERIES, sim)):
        order = row.argsort()[::-1]
        top = [(names[j], round(float(row[j]), 3)) for j in order[:3]]
        print(qi, "->", top)
        if auto:
            # 기대 장면 최고점과 다른 장면 최고점의 차이: 양수면 맞게 구분, 클수록 여유가 큼
            hit = [float(row[j]) for j in order if names[j].startswith(EXPECTED[i] + "_")]
            miss = [float(row[j]) for j in order if not names[j].startswith(EXPECTED[i] + "_")]
            margin = round(max(hit) - max(miss), 3)
            print(f"  기대 장면 {EXPECTED[i]}: 1위 {names[order[0]]}, 차이 {margin:+.3f}")
            rec.check(f"'{qi}' 1위", names[order[0]].startswith(EXPECTED[i] + "_"),
                      f"{top} margin={margin:+.3f}")
        elif i < len(args.judge):
            rec.check(f"'{qi}' 1위", args.judge[i].upper() == "Y", str(top))

    if not args.no_video and clip.exists():
        if args.video_frames:
            with tempfile.TemporaryDirectory() as tmp:
                frames, sec = video_frames(clip, tmp)
                with rec.time(f"동영상 {sec:.0f}초 (1 fps 프레임 {len(frames)}장)", file=clip.name,
                              n=len(frames), method="frames", batch=IMAGE_BATCH):
                    f_emb = model.encode([{"image": p} for p in frames], batch_size=IMAGE_BATCH,
                                         normalize_embeddings=True)
            # 프레임 평균을 다시 정규화해 동영상 하나의 벡터로 쓴다 (프레임 단위 검색 결과도 함께 출력)
            vid_emb = f_emb.mean(0)
            vid_emb = vid_emb / np.linalg.norm(vid_emb)
            fsim = q @ f_emb.T
            for qi, row in zip(QUERIES, fsim):
                print(f"  {qi} -> 가장 가까운 프레임 {int(row.argmax())}초 ({row.max():.3f})")
        else:
            with rec.time("동영상 1개", file=clip.name, method="video"):
                vid_emb = model.encode({"video": str(clip)}, normalize_embeddings=True)
        vsim = (q @ vid_emb).round(3)
        print("질의-동영상 유사도:", vsim)
        rec.row["video_scores"] = dict(zip(QUERIES, map(float, vsim)))

    if args.mixed:
        mixed = {"text": MIXED_TEXT, "image": [images[0]]}
        with rec.time("텍스트+이미지 혼합 1개", file=names[0]):
            m_emb = model.encode(mixed, normalize_embeddings=True)
        # 같은 이미지만 넣은 벡터와 얼마나 다른지, 질의와의 점수가 어떻게 바뀌는지 본다
        msim, isim = (q @ m_emb).round(3), (q @ img_emb[0]).round(3)
        print(f"혼합 입력 '{MIXED_TEXT}' + {names[0]}: 질의 유사도 {msim} (이미지만 {isim}), "
              f"이미지만 벡터와 코사인 {float(m_emb @ img_emb[0]):.3f}")
        rec.row["mixed"] = {"text": MIXED_TEXT, "image": names[0],
                            "query_scores": dict(zip(QUERIES, map(float, msim))),
                            "image_only_scores": dict(zip(QUERIES, map(float, isim))),
                            "cos_to_image_only": round(float(m_emb @ img_emb[0]), 3)}
    rec.save()


if __name__ == "__main__":
    main()
