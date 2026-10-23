"""2단계 설치 확인. torch 가 PyPI 판(GPU 미지원)으로 덮어쓰이지 않았는지 본다."""
import torch
import transformers
import sentence_transformers as st

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", st.__version__)
assert torch.cuda.is_available(), "Jetson torch 가 덮어쓰였는지 확인 (setup/02_install_torch.sh 재실행)"
major, minor = (int(x) for x in st.__version__.split(".")[:2])
assert (major, minor) >= (6, 1), "sentence-transformers 6.1.0 이상 필요 (prompt_name 지원)"
