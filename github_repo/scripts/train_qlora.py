# -*- coding: utf-8 -*-
"""
Quick SFT with QLoRA for CLTS (Qwen2-1.5B-Instruct)
- 4090 48G 目标：~4小时拿到一个能用的 LoRA 适配器
- 关键策略：4bit + LoRA + packing + 降序列 + 控制 max_steps + 8-bit 优化器
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ========= 1) 路径与可调参数（按需改） =========
# 本地模型（指向 snapshots 内含 model.safetensors 的那一层目录）
MODEL_ID = "/home/vipuser/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"

# 数据目录：优先用你抽样过的 subset；若还没抽样好，可改回 processed
DATA_DIR = "/home/vipuser/workshop/projects/clts-sum-qwen2/data/clts/processed/subset"
# DATA_DIR = "/home/vipuser/workshop/projects/clts-sum-qwen2/data/clts/processed"

OUT_DIR  = "/home/vipuser/workshop/projects/clts-sum-qwen2/runs/qwen2_1p5b_clts_qlora_fast"

# 子集上限（再次保险，可视情况调小提速）
MAX_TRAIN = 50000
MAX_EVAL  = 10000

# 训练“快跑”超参（按 4090 48G 估的安全值；OOM 就把 per_device_train_batch_size 降到 2）
MAX_LENGTH = 1024          # 序列长度（长文建议分块→二次汇总，别一把梭 2048）
PER_DEVICE_TRAIN_BS = 4    # 显存紧张就降到 2
GRAD_ACCUM = 2             # 有效 batch ≈ 8；显存紧张可把它加大（4/8）
MAX_STEPS = 3000           # 控制总步数；想更快先设 2000
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
LOG_STEPS = 50
SAVE_STEPS = 1000
USE_BF16 = True
USE_PACKING = True
USE_PAGED_ADAMW_8BIT = True

# 可选：是否启用梯度检查点（省显存、稍降速度）
ENABLE_GRADIENT_CHECKPOINTING = False

# ========= 2) 加速/数值稳定开关 =========
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ 建议开启，提速明显
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # 走稳定下载/本地

# ========= 3) 数据加载与格式化 =========
def load_chat_jsonl(tok, path):
    ds = load_dataset("json", data_files=path, split="train")
    # 将 messages 用 Qwen 的 chat template 渲染为纯文本
    def to_text(ex):
        text = tok.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    ds = ds.map(to_text, remove_columns=ds.column_names)
    # 只保留 text 一列，配合 packing 更省事
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds

def main():
    assert os.path.exists(MODEL_ID), f"模型目录不存在：{MODEL_ID}"
    assert os.path.exists(DATA_DIR), f"数据目录不存在：{DATA_DIR}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)

    # ---- 4bit + nf4 量化配置 ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # ---- 基座模型（可选：attn_implementation="flash_attention_2" 若已安装 flash-attn）----
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb,
        dtype=torch.bfloat16,
        local_files_only=True,
        # attn_implementation="flash_attention_2",
    )

    if ENABLE_GRADIENT_CHECKPOINTING:
        base.gradient_checkpointing_enable()

    # ---- LoRA 适配配置（Qwen2 常见模块）----
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora)

    # ---- 数据集 ----
    train_path = os.path.join(DATA_DIR, "train_chat_50k.jsonl")  # 优先用抽样文件名
    valid_path = os.path.join(DATA_DIR, "valid_chat_10k.jsonl")
    # 若找不到抽样文件，则回退到原始命名
    if not os.path.exists(train_path):
        train_path = os.path.join(DATA_DIR, "train_chat.jsonl")
    if not os.path.exists(valid_path):
        valid_path = os.path.join(DATA_DIR, "valid_chat.jsonl")

    train_ds = load_chat_jsonl(tok, train_path)
    eval_ds  = load_chat_jsonl(tok, valid_path) if os.path.exists(valid_path) else None

    # 再裁一次上限（进一步控时）
    if MAX_TRAIN and len(train_ds) > MAX_TRAIN:
        train_ds = train_ds.select(range(MAX_TRAIN))
    if eval_ds is not None and MAX_EVAL and len(eval_ds) > MAX_EVAL:
        eval_ds = eval_ds.select(range(MAX_EVAL))

    # ---- 训练配置（按“步数”而非“epoch”）----
    optim_name = "paged_adamw_8bit" if USE_PAGED_ADAMW_8BIT else "adamw_torch"
    conf = SFTConfig(
        output_dir=OUT_DIR,
        max_length=MAX_LENGTH,
        packing=USE_PACKING,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_steps=MAX_STEPS,
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        bf16=USE_BF16,
        optim=optim_name,
        # 轻量起见：不做中途评估（最快）。如需评估，可增加 eval_strategy/steps（取决于你本地 trl 版本）
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,   # None 则无评估
        args=conf,
        # 注：你当前的 TRL 版本不接受 tokenizer/dataset_text_field 等额外参数
    )

    # ---- 开始训练 ----
    trainer.train()

    # ---- 保存 LoRA 适配器 ----
    adapter_dir = os.path.join(OUT_DIR, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    print(f"[OK] LoRA adapter saved to: {adapter_dir}")

if __name__ == "__main__":
    main()

