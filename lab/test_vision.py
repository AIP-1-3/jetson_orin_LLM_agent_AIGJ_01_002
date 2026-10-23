"""4단계 영상 테스트: 텍스트 질의로 이미지·동영상 찾기.

자료(media/)는 scripts/record_clip.sh(OpenCV 녹화)로 IP 카메라에서 만들거나 직접 복사한다.
정답 확인은 사람이 상위 3개 이미지를 눈으로 보고 판단한다. 판정은 --judge 로 남길 수 있다.
"""
import argparse
import glob

from bench import Recorder, media_dir
from eg2 import config_name, load

QUERIES = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="동영상 디코딩이 실패할 때 이미지만 검색")
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
    with rec.time(f"이미지 {len(images)}장", n=len(images)):
        img_emb = model.encode(inputs, normalize_embeddings=True)

    q = model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
    sim = q @ img_emb.T
    for i, (qi, row) in enumerate(zip(QUERIES, sim)):
        order = row.argsort()[::-1][:3]
        top = [(images[j].split("/")[-1], round(float(row[j]), 3)) for j in order]
        print(qi, "->", top)
        if i < len(args.judge):
            rec.check(f"'{qi}' 1위", args.judge[i].upper() == "Y", str(top))

    if not args.no_video and clip.exists():
        with rec.time("동영상 1개", file=clip.name):
            vid_emb = model.encode({"video": str(clip)}, normalize_embeddings=True)
        print("질의-동영상 유사도:", (q @ vid_emb).round(3))
    rec.save()


if __name__ == "__main__":
    main()
