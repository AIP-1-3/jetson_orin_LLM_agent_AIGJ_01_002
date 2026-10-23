"""10-1 렌즈 보정: 체커보드를 여러 각도로 비추면 자동으로 20장을 모아 K(내부 행렬), dist(왜곡)를 저장
화면이 없어도 동작한다 (진행 상황은 터미널, 마지막 검출 이미지는 calib/camN_last.jpg)
사용법: python calib_intrinsics.py --name cam0 --src rtsp://...   (2호기 VIGI)
        python calib_intrinsics.py --name cam1 --src http://<1호기>:8091/stream.mjpg   (1호기 USB, cam_server.py)
자료와 다른 점: 장치 번호 대신 영상 주소(--src)를 받고, 결과 파일 이름은 --name 으로 정한다.
IP 카메라는 해상도를 바꿀 수 없어 실제 프레임 크기를 그대로 저장한다 (VIGI 보조 스트림 848x480).
"""
import argparse, os, time
import cv2, numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
from camera import LatestFrame, mask   # RTSP(2호기 VIGI)·USB·HTTP MJPEG(1호기 cam_server.py)를 같은 방식으로 연다

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="cam0 또는 cam1 (proximity_app 의 순서)")
ap.add_argument("--src", required=True, help="RTSP·HTTP 주소 또는 USB 번호")
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))   # USB 일 때만 요청 해상도로 쓴다
os.makedirs("calib", exist_ok=True)

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

cam = LatestFrame(args.src, W, H)
print("소스:", mask(args.src))

obj_pts, img_pts, last_t, last_c = [], [], 0.0, None
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
while len(img_pts) < args.n:
    frame, _, _ = cam.read()
    if frame is None:
        time.sleep(0.05); continue
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), cv2.CALIB_CB_FAST_CHECK)
    if not found or time.time() - last_t < 1.0:
        continue
    c = corners.reshape(-1, 2).mean(0)
    if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치는 건너뜀
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp); img_pts.append(corners)
    last_t, last_c = time.time(), c
    cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)
    cv2.imwrite(f"calib/{args.name}_last.jpg", frame)
    print(f"  {len(img_pts)}/{args.n}")

rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None)
np.savez(f"calib/{args.name}_intr.npz", K=K, dist=dist, res=np.array([W, H]), rms=rms)
cam.close()
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/{args.name}_intr.npz ({W}x{H})")
