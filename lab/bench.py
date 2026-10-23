"""측정값을 측정 조건과 함께 results/*.jsonl 에 남긴다.

보고서의 모든 숫자가 이 파일에서 나오게 해서, 문서·발표·코드의 숫자가 어긋나지 않게 한다.
Jetson 은 CPU·GPU 가 RAM 을 공유하므로 프로세스 RSS 만으로는 GPU 사용량이 안 보인다.
그래서 시스템 전체 사용량(MemTotal - MemAvailable)과 torch 할당량을 함께 기록한다.
"""
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def _meminfo_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            info[key] = int(val.split()[0]) // 1024
    return info


def system_used_mb():
    m = _meminfo_mb()
    return m["MemTotal"] - m["MemAvailable"]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def conditions():
    """측정 조건: 장비, 전력 모드, L4T, 라이브러리 버전, 커밋."""
    import torch  # 지연 import: bench 만 쓰는 스크립트가 torch 로드 비용을 내지 않게
    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text().rstrip("\0")
    except OSError:
        pass
    power = _run(["nvpmodel", "-q"]).splitlines()
    try:
        import sentence_transformers as st
        st_ver = st.__version__
    except ImportError:
        st_ver = None
    return {
        "device": model or platform.node(),
        "l4t": _run(["head", "-n1", "/etc/nv_tegra_release"]),
        "power_mode": power[0] if power else "unknown (sudo nvpmodel -q 로 확인)",
        "torch": torch.__version__,
        "sentence_transformers": st_ver,
        "commit": _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
    }


class Recorder:
    """한 스크립트 실행의 측정값을 모아 results/<step>.jsonl 에 한 줄로 추가한다."""

    def __init__(self, step, config, dtype="bfloat16"):
        self.row = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": step,
            "config": config,
            "dtype": dtype,
            "ram_before_load_mb": system_used_mb(),
            "measures": {},
            "checks": {},
        }

    def loaded(self):
        """모델 로드 직후 호출. 로드 후 RAM 을 기록한다."""
        import torch
        torch.cuda.synchronize()
        self.row["ram_after_load_mb"] = system_used_mb()
        self.row["torch_alloc_after_load_mb"] = torch.cuda.memory_allocated() // 2**20

    @contextmanager
    def time(self, name, **meta):
        """with rec.time("질의 2개"): ... 블록의 지연(ms)을 잰다. GPU 비동기 실행을 기다린 뒤 끝낸다."""
        import torch
        torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        self.row["measures"].setdefault(name, []).append({"ms": round(ms, 1), **meta})
        print(f"  [{name}] {ms:.0f} ms")

    def check(self, name, passed, detail=""):
        """'질의가 맞는 문서를 1위로 찾았는가' 같은 정답 여부를 남긴다."""
        self.row["checks"][name] = {"pass": bool(passed), "detail": detail}
        print(f"  [{'OK' if passed else 'NG'}] {name} {detail}")

    def save(self):
        import torch
        self.row["ram_peak_mb"] = system_used_mb()
        self.row["torch_peak_alloc_mb"] = torch.cuda.max_memory_allocated() // 2**20
        self.row["conditions"] = conditions()
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{self.row['step']}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.row, ensure_ascii=False) + "\n")
        print(f"기록: {path.relative_to(REPO)}")


def media_dir():
    """테스트 자료 폴더. 사람이 찍힐 수 있어 git 에는 올리지 않는다 (.gitignore)."""
    return Path(os.environ.get("EG2_MEDIA", REPO / "media"))
