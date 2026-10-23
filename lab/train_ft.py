"""7단계 미세 조정: data/train.jsonl 로 텍스트 모델(270M)을 추가 학습한다 (자료 7단계 train_ft.py).

8GB 공유 메모리라 텍스트 인코더만 올리고, gradient checkpointing 을 내부 HF 모델에 직접 켠다
(Trainer 인자로 켜면 every_n_layers TypeError, 자료 Q9). 학습 시간을 results/07_finetune.jsonl 에 남긴다.
사용: nohup python train_ft.py > ../results/logs/07_train.log 2>&1 &
      OOM 이면 BATCH=4 GRAD_ACC=4 python train_ft.py

2호기 첫 시도는 배치 8×2, 4×4 모두 두 번째 step 에서 OOM 이었다 (README 문제 해결 기록 11).
데스크톱이 이미 약 4.3 GB 를 쓰고 있어 학습에 남은 메모리가 약 3 GB 였다. 그래서 두 가지를 기본으로 켠다.
- FREEZE_EMB=1: 토큰 임베딩 표를 얼린다. 모델 파라미터 271.0M 중 134.2M(49.5%)이 이 표라서,
  얼리면 기울기·옵티마이저 상태가 그만큼 줄어든다. 도메인 문장의 단어는 이미 어휘에 있어 표를 고칠 필요가 작다.
- MAX_LEN=128: 학습 문장 최대 길이. 데이터의 질의·문서는 프롬프트를 붙여도 이보다 짧다.
"""
import os
import time

# 오류 메시지가 권하는 단편화 완화. torch import 전에 정해야 적용된다
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from bench import REPO, Recorder
from eval_ir import DATA, build_evaluator

OUT = REPO / "models" / os.environ.get("OUT_NAME", "eg2-classroom")   # epoch 비교 때 이름을 바꿔 덮어쓰지 않게
BATCH = int(os.environ.get("BATCH", 8))
GRAD_ACC = int(os.environ.get("GRAD_ACC", 2))
EPOCHS = float(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-5))
FREEZE_EMB = os.environ.get("FREEZE_EMB", "1") == "1"
MAX_LEN = int(os.environ.get("MAX_LEN", 128))


def main():
    rec = Recorder("07_finetune", f"학습 (배치 {BATCH}×{GRAD_ACC}, {EPOCHS:g} epoch, lr {LR:g}"
                   + (", 임베딩 표 고정" if FREEZE_EMB else "") + f", 최대 {MAX_LEN} 토큰)")
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None, "audio_config": None})
    hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.max_seq_length = MAX_LEN
    if FREEZE_EMB:
        hf.get_input_embeddings().requires_grad_(False)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rec.row["params"] = {"total_m": round(n_all / 1e6, 1), "trainable_m": round(n_train / 1e6, 1)}
    print(f"파라미터 {n_all / 1e6:.1f}M 중 학습 {n_train / 1e6:.1f}M")
    rec.loaded()

    train = load_dataset("json", data_files=str(DATA / "train.jsonl"), split="train")
    # MatryoshkaLoss 로 감싸야 학습 후에도 256/128차원 절단이 유지된다
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=[768, 512, 256, 128])
    args = SentenceTransformerTrainingArguments(
        output_dir=str(REPO / "runs" / "eg2-classroom"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR, warmup_steps=0.1,      # Transformers v5: 비율을 float 로
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # 배치 안 중복 문서 방지 (같은 문서가 오답으로 쓰이지 않게)
        prompts={"anchor": model.prompts["SearchQuery"], "positive": model.prompts["Document"]},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=10, report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                         loss=loss, evaluator=build_evaluator())
    t = time.perf_counter()
    out = trainer.train()
    rec.row["train_seconds"] = round(time.perf_counter() - t, 1)
    rec.row["train_pairs"] = len(train)
    rec.row["train_loss"] = round(out.training_loss, 4)
    rec.row["log_history"] = [h for h in trainer.state.log_history if "loss" in h or "eval_classroom_cosine_ndcg@10" in h]
    model.save_pretrained(str(OUT))
    print("저장:", OUT)
    rec.save()


if __name__ == "__main__":
    main()
