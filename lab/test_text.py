"""3단계 텍스트 테스트: 한국어 질의가 맞는 문서를 1위로 찾는지, 차원(768/256/128)별 지연과 점수.

--no-prompt 로 실행하면 SearchQuery/Document 프롬프트 없이 같은 비교를 한다 (자료의 변형 과제).
--sustain 초 를 주면 dim=768 질의를 그 시간 동안 반복해 처리량(질의/초)을 잰다.
한 번 추론은 약 150 ms 라서 1초 간격 tegrastats 에 GPU 사용률이 거의 잡히지 않기 때문이다.
"""
import argparse
import time

from bench import Recorder
from eg2 import config_name, load

DOCS = [
    "환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
    "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
    "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
    "Jetson Orin Nano는 67 TOPS의 AI 성능을 제공한다.",
]
# (질의, 정답 문서 번호)
QUERIES = [("수업 중에 졸리고 머리가 맑지 않아요", 0), ("여름에 냉방 몇 도로 맞춰야 하나요?", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prompt", action="store_true", help="프롬프트 없이 인코딩 (변형 과제)")
    ap.add_argument("--sustain", type=float, default=0, metavar="초",
                    help="dim=768 질의를 이 시간 동안 반복해 처리량 측정 (tegrastats 로 GPU 사용률 확인용)")
    args = ap.parse_args()
    qp, dp = (None, None) if args.no_prompt else ("SearchQuery", "Document")

    rec = Recorder("03_text", config_name() + (" / 프롬프트 없음" if args.no_prompt else ""))
    model = load()
    rec.loaded()

    d = model.encode(DOCS, prompt_name=dp, normalize_embeddings=True)
    queries = [q for q, _ in QUERIES]

    for dim in [768, 256, 128]:
        # 첫 실행은 CUDA 초기화로 느리다. 한 번 데우고 두 번째 값을 기록한다 (자료 6단계 기준)
        model.encode(queries, prompt_name=qp, truncate_dim=dim, normalize_embeddings=True)
        print(f"\n[dim={dim}]")
        with rec.time(f"질의 2개 dim={dim}", dim=dim):
            q = model.encode(queries, prompt_name=qp, truncate_dim=dim, normalize_embeddings=True)
        dd = d[:, :dim]
        dd = dd / (dd ** 2).sum(1, keepdims=True) ** 0.5   # 자른 뒤 재정규화
        sim = q @ dd.T
        for (qi, answer), row in zip(QUERIES, sim):
            best = int(row.argmax())
            rec.check(f"dim={dim} '{qi}'", best == answer,
                      f"-> {DOCS[best][:20]}... ({row[best]:.3f}, 정답 점수 {row[answer]:.3f})")
    if args.sustain > 0:
        n, t0 = 0, time.perf_counter()
        with rec.time(f"반복 {args.sustain:.0f}초 dim=768", seconds=args.sustain):
            while time.perf_counter() - t0 < args.sustain:
                model.encode(queries, prompt_name=qp, normalize_embeddings=True)
                n += len(queries)
        qps = n / (time.perf_counter() - t0)
        rec.row["throughput_qps"] = round(qps, 1)
        print(f"  처리량 {qps:.1f} 질의/초 ({n}개)")
    rec.save()


if __name__ == "__main__":
    main()
