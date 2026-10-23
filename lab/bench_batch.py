"""6단계 추가 실험: 묶음(batch) 크기에 따른 텍스트 처리량.

3단계 --sustain 은 질의 2개씩이라 GPU 사용률이 10~13% 에 그쳤다.
같은 시간 동안 묶음 크기만 바꿔 처리량(질의/초)과 GPU 사용률이 어떻게 변하는지 잰다.
tegrastats 로그와 시각을 맞춰 보도록 묶음마다 시작·끝 시각을 출력한다.
사용: TS_MS=200 ./scripts/tegrastats_log.sh 06_batch  (다른 창)
      python bench_batch.py [--sizes 1 8 32] [--seconds 10]
"""
import argparse
import time
from datetime import datetime

from bench import Recorder
from eg2 import config_name, load
from test_text import DOCS, QUERIES

# 실제 질의 문장을 돌려 써서 길이 분포를 3단계와 같게 둔다
POOL = [q for q, _ in QUERIES] + DOCS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args()

    rec = Recorder("06_batch", config_name())
    model = load()
    rec.loaded()

    for bs in args.sizes:
        batch = [POOL[i % len(POOL)] for i in range(bs)]
        model.encode(batch, prompt_name="SearchQuery", batch_size=bs)   # 데우기
        n, t0 = 0, time.perf_counter()
        start = datetime.now().strftime("%H:%M:%S")
        with rec.time(f"묶음 {bs}", batch=bs, seconds=args.seconds):
            while time.perf_counter() - t0 < args.seconds:
                model.encode(batch, prompt_name="SearchQuery", batch_size=bs, normalize_embeddings=True)
                n += bs
        took = time.perf_counter() - t0
        qps = n / took
        rec.row["measures"][f"묶음 {bs}"][-1].update({"queries": n, "qps": round(qps, 1),
                                                      "ms_per_batch": round(took * 1000 * bs / n, 1)})
        print(f"  묶음 {bs:>3}: {qps:7.1f} 질의/초, 묶음당 {took * 1000 * bs / n:.0f} ms "
              f"({start}~{datetime.now().strftime('%H:%M:%S')})")
        time.sleep(2)   # tegrastats 에서 구간이 구분되게 쉰다
    rec.save()


if __name__ == "__main__":
    main()
