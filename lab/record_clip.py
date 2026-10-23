"""4단계 자료 만들기: IP 카메라(.env 의 CAM_URL)에서 N초 녹화 + 2초마다 프레임 저장.

nvidia-jetpack 이 설치한 NVIDIA 판 ffmpeg 는 네트워크 프로토콜과 libx264 가 빠져 있어
자료의 `ffmpeg -i rtsp://...` 녹화가 되지 않는다. OpenCV 는 자체 FFmpeg 라이브러리로 RTSP 를 열 수 있어서
camera.open_capture() 로 받아 OpenCV 로 저장한다.
사용: python record_clip.py [초]   (기본 10초)
"""
import sys
import time

import cv2

from bench import media_dir
from camera import default_source, mask, open_capture

FRAME_EVERY_S = 2.0   # 자료와 같이 2초마다 한 장 (ffmpeg fps=0.5)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    src = default_source()
    if src == "0":
        raise SystemExit(".env 의 CAM_URL 이 없습니다: cp .env.example .env 후 채우세요")
    media = media_dir()
    media.mkdir(parents=True, exist_ok=True)
    for old in media.glob("frame_*.jpg"):   # 이전 녹화 프레임과 섞이지 않게
        old.unlink()

    print("소스:", mask(src))
    cap = open_capture(src)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("첫 프레임을 받지 못했습니다 (주소·계정·네트워크 확인)")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 60 else 15.0   # 카메라가 fps 를 안 알려 주면 15 로 가정
    # mp4v: 추가 코덱 없이 OpenCV 기본 빌드로 쓸 수 있고, test_vision 의 PyAV 가 읽을 수 있다
    out = cv2.VideoWriter(str(media / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    t0 = time.time()
    n, saved, next_shot = 0, 0, 0.0
    while ok and time.time() - t0 < seconds:
        out.write(frame)
        n += 1
        if time.time() - t0 >= next_shot:
            saved += 1
            cv2.imwrite(str(media / f"frame_{saved:02d}.jpg"), frame)
            next_shot += FRAME_EVERY_S
        ok, frame = cap.read()
    out.release()
    cap.release()
    took = time.time() - t0
    print(f"clip.mp4: {w}x{h}, {n}프레임 / {took:.1f}초 (받은 fps {n / took:.1f}), 프레임 {saved}장")


if __name__ == "__main__":
    main()
