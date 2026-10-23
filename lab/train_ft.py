"""7단계 미세 조정: data/train.jsonl 로 텍스트 모델(270M)을 추가 학습한다 (자료 7단계 train_ft.py).

8GB 공유 메모리라 텍스트 인코더만 올리고, gradient checkpointing 을 내부 HF 모델에 직접 켠다
(Trainer 인자로 켜면 every_n_layers TypeError, 자료 Q9). 학습 시간을 results/07_finetune.jsonl 에 남긴다.
사용: nohup python train_ft.py > ../results/logs/07_train.log 2>&1 &
      OOM 이면 BATCH=4 GRAD_ACC=4 python train_ft.py
"""
import os
import time

import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from bench import REPO, Recorder
from eval_ir import DATA, build_evaluator

OUT = REPO / "models" / "eg2-classroom"
BATCH = int(os.environ.get("BATCH", 8))
GRAD_ACC = int(os.environ.get("GRAD_ACC", 2))
EPOCHS = float(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-5))


def main():
    rec = Recorder("07_finetune", f"학습 (배치 {BATCH}×{GRAD_ACC}, {EPOCHS:g} epoch, lr {LR:g})")
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None, "audio_config": None})
    hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
