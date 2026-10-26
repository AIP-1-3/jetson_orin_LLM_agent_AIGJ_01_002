"""6단계 성능 표 만들기 (3~5단계 + 전체 모델 혼합 06_all): results/*.jsonl 과 tegrastats 로그를 읽어 Markdown 표로 출력한다.

보고서 숫자를 손으로 옮기지 않고 이 출력에서 그대로 붙여 넣는다 (문서·코드 숫자 일치).
사용: python lab/results_table.py            # 3~5단계 최신 측정
      python lab/results_table.py --all      # 모든 측정 이력
"""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
KST = timezone(timedelta(hours=9))   # 장비 시계 (로그 파일 이름)
STEPS = ["03_text", "04_vision", "05_audio", "06_all"]


def load_rows(step):
    path = RESULTS / f"{step}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _log_matches(log, row_time):
    """로그 이름의 시작 시각(장비 시간, KST)이 측정 기록 시각보다 앞이고 30분 안쪽일 때만 짝으로 본다.

    tegrastats 없이 다시 잰 기록(예: 3단계 --no-prompt)에 옛 로그의 최대 RAM 이 붙어
    '최대 RAM < 로드 후 RAM' 처럼 어긋난 표가 나온 적이 있다 (2026-10-26).
    """
    m = re.search(r"_(\d{8}-\d{6})\.log$", log.name)
    if not m or not row_time:
        return True
    start = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=KST)
    end = datetime.fromisoformat(row_time)
    return timedelta(0) <= end - start <= timedelta(minutes=30)


def tegrastats_summary(step, row_time=None):
    """측정 기록과 짝이 맞는 가장 최근 로그에서 RAM 최대값(MB)과 GPU(GR3D) 사용률 평균·최대를 뽑는다."""
    logs = [p for p in sorted((RESULTS / "logs").glob(f"{step}_*.log")) if _log_matches(p, row_time)]
    if not logs:
        return None
    ram, gpu, power = [], [], []
    for line in logs[-1].read_text(errors="ignore").splitlines():
        if m := re.search(r"RAM (\d+)/\d+MB", line):
            ram.append(int(m.group(1)))
        if m := re.search(r"GR3D_FREQ (\d+)%", line):
            gpu.append(int(m.group(1)))
        if m := re.search(r"VDD_IN (\d+)mW", line):   # 보드 전체 입력 전력(현재값)
            power.append(int(m.group(1)))
    if not ram:
        return None
    return {"log": logs[-1].relative_to(REPO).as_posix(), "ram_max": max(ram),
            "gpu_avg": round(sum(gpu) / len(gpu)) if gpu else None,
            "gpu_max": max(gpu) if gpu else None,
            "power_avg": round(sum(power) / len(power)) if power else None,
            "power_max": max(power) if power else None}


def _latency(m, row):
    """반복 측정(--sustain)은 구간 전체 시간이라 지연 대신 처리량을 보여 준다."""
    if "seconds" in m and "throughput_qps" in row:
        return f"처리량 {row['throughput_qps']} 질의/초 ({m['seconds']:.0f}초)"
    return m["ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    print("| 구성 | 입력 | 실행 전 RAM (MB) | 로드 후 RAM (MB) | 모델 증가분 (MB) | 최대 RAM (MB) | 지연 (ms) | GPU 사용률 평균/최대 (%) | 전력 평균/최대 (mW) | 정답 | 커밋 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cond = None
    for step in STEPS:
        rows = load_rows(step)
        rows = rows if args.all else rows[-1:]
        for r in rows:
            ts = tegrastats_summary(step, r.get("time"))
            cond = r["conditions"]
            ok = sum(c["pass"] for c in r["checks"].values())
            total = len(r["checks"])
            gpu = f"{ts['gpu_avg']}/{ts['gpu_max']}" if ts and ts["gpu_avg"] is not None else "-"
            peak = ts["ram_max"] if ts else r["ram_peak_mb"]
            pw = f"{ts['power_avg']}/{ts['power_max']}" if ts and ts["power_avg"] is not None else "-"
            before, after = r["ram_before_load_mb"], r.get("ram_after_load_mb")
            delta = after - before if after is not None else "-"
            for name, ms in r["measures"].items():
                print(f"| {r['config']} | {name} | {before} | {after if after is not None else '-'} | {delta} | {peak} | "
                      f"{_latency(ms[-1], r)} | {gpu} | {pw} | {ok}/{total} | {cond['commit']} |")
    if cond:
        print(f"\n측정 조건: {cond['device']}, {cond['power_mode']}, {cond['l4t']}, "
              f"torch {cond['torch']}, sentence-transformers {cond['sentence_transformers']}, bfloat16")
        print("RAM 은 시스템 전체 사용량(MemTotal - MemAvailable). 다른 프로세스가 쓰는 양이 섞이므로 모델 몫은 '모델 증가분'(로드 후 - 실행 전)으로 본다. tegrastats 로그가 있으면 최대 RAM·GPU·전력은 로그 전체 구간(모델 로드 포함) 값.")
    else:
        print("\n측정 기록이 없습니다. lab/test_*.py 를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
