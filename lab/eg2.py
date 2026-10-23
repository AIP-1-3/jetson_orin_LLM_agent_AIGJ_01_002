"""공통 로더 (자료 2단계). 필요한 인코더만 골라 올려 8GB 공유 메모리를 아낀다."""
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
# float16 은 임베딩에 NaN 을 만든다 (자료 6단계). bfloat16 을 기본으로 둔다.
DTYPE = torch.bfloat16


def load(vision=False, audio=False):
    cfg = {}
    if not vision:
        cfg["vision_config"] = None   # 미사용 인코더는 로드하지 않음
    if not audio:
        cfg["audio_config"] = None
    return SentenceTransformer(
        MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": DTYPE},
        config_kwargs=cfg,
    )


def config_name(vision=False, audio=False):
    """측정 표의 '구성' 열 이름. 자료 6단계 표와 같은 이름을 쓴다."""
    if vision and audio:
        return "전체 (740M)"
    if vision:
        return "+비전 (440M)"
    if audio:
        return "+오디오 (570M)"
    return "텍스트 전용 (270M)"
