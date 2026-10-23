"""6단계 성능 표 만들기: results/*.jsonl 과 tegrastats 로그를 읽어 Markdown 표로 출력한다.

보고서 숫자를 손으로 옮기지 않고 이 출력에서 그대로 붙여 넣는다 (문서·코드 숫자 일치).
사용: python lab/results_table.py            # 3~5단계 최신 측정
      python lab/results_table.py --all      # 모든 측정 이력
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
STEPS = ["03_text", "04_vision", "05_audio"]


def load_rows(step):
    path = RESULTS / f"{step}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tegrastats_summary(step):
    """가장 최근 로그에서 RAM 최대값(MB)과 GPU(GR3D) 사용률 평균·최대를 뽑는다."""
    logs = sorted((RESULTS / "logs").glob(f"{step}_*.log"))
    if not logs:
        return None
    ram, gpu = [], []
    for line in logs[-1].read_text(errors="ignore").splitlines():
        if m := re.search(r"RAM (\d+)/\d+MB", line):
            ram.append(int(m.group(1)))
        if m := re.search(r"GR3D_FREQ (\d+)%", line):
            gpu.append(int(m.group(1)))
    if not ram:
        return None
    return {"log": logs[-1].relative_to(REPO).as_posix(), "ram_max": max(ram),
            "gpu_avg": round(sum(gpu) / len(gpu)) if gpu else None,
            "gpu_max": max(gpu) if gpu else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    print("| 구성 | 입력 | 실행 전 RAM (MB) | 로드 후 RAM (MB) | 모델 증가분 (MB) | 최대 RAM (MB) | 지연 (ms) | GPU 사용률 평균/최대 (%) | 정답 | 커밋 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cond = None
    for step in STEPS:
        rows = load_rows(step)
        rows = rows if args.all else rows[-1:]
        ts = tegrastats_summary(step)
        for r in rows:
            cond = r["conditions"]
            ok = sum(c["pass"] for c in r["checks"].values())
            total = len(r["checks"])
            gpu = f"{ts['gpu_avg']}/{ts['gpu_max']}" if ts and ts["gpu_avg"] is not None else "-"
            peak = ts["ram_max"] if ts else r["ram_peak_mb"]
            before, after = r["ram_before_load_mb"], r.get("ram_after_load_mb")
            delta = after - before if after is not None else "-"
            for name, ms in r["measures"].items():
                print(f"| {r['config']} | {name} | {before} | {after if after is not None else '-'} | {delta} | {peak} | "
                      f"{ms[-1]['ms']} | {gpu} | {ok}/{total} | {cond['commit']} |")
    if cond:
        print(f"\n측정 조건: {cond['device']}, {cond['power_mode']}, {cond['l4t']}, "
              f"torch {cond['torch']}, sentence-transformers {cond['sentence_transformers']}, bfloat16")
        print("RAM 은 시스템 전체 사용량(MemTotal - MemAvailable). 다른 프로세스가 쓰는 양이 섞이므로 모델 몫은 '모델 증가분'(로드 후 - 실행 전)으로 본다. tegrastats 로그가 있으면 최대 RAM 은 로그 값.")
    else:
        print("\n측정 기록이 없습니다. lab/test_*.py 를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
