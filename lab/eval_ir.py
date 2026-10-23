"""7단계 평가: data/eval.jsonl 로 검색 정확도(accuracy@1·3, nDCG@10)를 잰다. 학습 전후 모두 쓴다.

자료 7단계 eval_ir.py 와 같은 평가기를 쓰고, 결과를 results/07_finetune.jsonl 에 측정 조건과 함께 남긴다.
사용: python eval_ir.py                          # 기준 모델
      python eval_ir.py ../models/eg2-classroom   # 미세 조정 모델
"""
import json
import sys
import time

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from bench import REPO, Recorder

DATA = REPO / "data"


def build_evaluator(path=DATA / "eval.jsonl", name="classroom"):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    corpus, queries, relevant, doc_id = {}, {}, {}, {}
    for i, r in enumerate(rows):
        d = doc_id.setdefault(r["positive"], f"d{len(doc_id)}")
        corpus[d] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        relevant.setdefault(f"q{i}", set()).add(d)
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name,
        query_prompt_name="SearchQuery", corpus_prompt_name="Document",
        accuracy_at_k=[1, 3, 5], truncate_dim=None)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
    label = "기준 (google/embeddinggemma-2)" if path == "google/embeddinggemma-2" else f"미세 조정 ({path.rstrip('/').split('/')[-1]})"
    rec = Recorder("07_finetune", label)
    model = SentenceTransformer(path, device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None, "audio_config": None})
    rec.loaded()
    t = time.perf_counter()
    res = build_evaluator()(model)
    rec.row["eval_seconds"] = round(time.perf_counter() - t, 1)
    for k in ["accuracy@1", "accuracy@3", "accuracy@5", "ndcg@10", "mrr@10"]:
        v = round(res[f"classroom_cosine_{k}"], 3)
        rec.row.setdefault("scores", {})[k] = v
        print(k, v)
    rec.save()


if __name__ == "__main__":
    main()
