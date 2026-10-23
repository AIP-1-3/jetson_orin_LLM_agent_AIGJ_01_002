"""7단계 확인 포인트: 미세 조정 뒤 학습 범위 밖 질의를 여전히 맞히는지 본다 (자료: "도메인 밖 일반 질의도 다시 돌려 본다").

두 묶음을 같은 방식(질의마다 정답 문서가 1위인지, 768/256/128 차원)으로 잰다.
- 3단계: test_text.py 의 문서·질의 그대로. 강의실 주제지만 학습 데이터와 문장이 다르다.
- 일반: 강의실과 무관한 생활·지식 문장 6쌍. 학습 데이터에 없는 주제라 망각 여부를 더 직접 보여 준다.
결과는 results/07_general.jsonl 에 모델별로 한 줄씩 남는다.

사용: python check_general.py                          # 기준 모델
      python check_general.py ../models/eg2-classroom   # 미세 조정 모델
"""
import sys

import torch
from sentence_transformers import SentenceTransformer

from bench import Recorder
from test_text import DOCS as CLASS_DOCS, QUERIES as CLASS_QUERIES

GEN_DOCS = [
    "김치찌개는 잘 익은 김치와 돼지고기를 넣고 끓이는 한국의 대표 찌개다.",
    "지하철 2호선은 서울 도심을 순환하는 노선으로 시청, 강남, 홍대입구 등을 지난다.",
    "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정이다.",
    "마라톤의 공식 거리는 42.195km 이다.",
    "고양이는 하루에 12~16시간 정도 잠을 잔다.",
    "파이썬에서 리스트를 정렬하려면 sort() 메서드나 sorted() 함수를 쓴다.",
]
GEN_QUERIES = [("저녁에 얼큰한 찌개 먹고 싶은데 뭐가 좋아?", 0), ("강남 가려면 몇 호선 타야 돼?", 1),
               ("식물은 어떻게 양분을 만들어?", 2), ("풀코스 달리기는 몇 킬로야?", 3),
               ("우리 집 고양이가 너무 많이 자요", 4), ("파이썬 배열 순서대로 늘어놓는 법", 5)]


def run(model, rec, name, docs, queries):
    d_full = model.encode(docs, prompt_name="Document", normalize_embeddings=True)
    ok = 0
    for dim in [768, 256, 128]:
        q = model.encode([x for x, _ in queries], prompt_name="SearchQuery", truncate_dim=dim, normalize_embeddings=True)
        d = d_full[:, :dim]
        d = d / (d ** 2).sum(1, keepdims=True) ** 0.5   # 자른 뒤 재정규화
        for (qi, answer), row in zip(queries, q @ d.T):
            best = int(row.argmax())
            ok += best == answer
            # 정답 점수와 정답이 아닌 문서 중 최고 점수의 차이: 클수록 여유 있게 맞힘
            margin = float(row[answer] - max(row[j] for j in range(len(docs)) if j != answer))
            rec.check(f"{name} dim={dim} '{qi}'", best == answer,
                      f"-> {docs[best][:16]}... ({row[best]:.3f}, 정답 {row[answer]:.3f}, 차이 {margin:+.3f})")
    print(f"{name}: {ok}/{len(queries) * 3}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
    rec = Recorder("07_general", "기준 (google/embeddinggemma-2)" if path.startswith("google/") else f"미세 조정 ({path.rstrip('/').split('/')[-1]})")
    model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None, "audio_config": None})
    rec.loaded()
    with torch.inference_mode():
        run(model, rec, "3단계", CLASS_DOCS, CLASS_QUERIES)
        run(model, rec, "일반", GEN_DOCS, GEN_QUERIES)
    rec.save()


if __name__ == "__main__":
    main()
