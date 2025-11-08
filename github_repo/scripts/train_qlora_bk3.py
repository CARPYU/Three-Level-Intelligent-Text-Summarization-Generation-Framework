import torch
import os  # 导入 os 模块，解决路径检查错误
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def load_chat_jsonl(tok, path):
    # 加载 jsonl 格式数据
    ds = load_dataset("json", data_files=path, split="train")
    # 将数据转为模型所需格式
    def to_text(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds

def main():
    # 本地模型路径（缓存目录）
    model_id = "/home/vipuser/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"  # 本地缓存路径
  # 本地缓存路径
    data_dir = "/home/vipuser/workshop/projects/clts-sum-qwen2/data/clts/processed"  # CLTS 数据路径
    out_dir = "/home/vipuser/workshop/projects/clts-sum-qwen2/runs/qwen2_1p5b_clts_qlora"  # 输出路径

    # 检查数据目录是否存在
    assert data_dir and os.path.exists(data_dir), "Please set DATA_DIR to processed jsonl dir"

    # 加载 tokenizer 和模型
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
        quantization_config=bnb, torch_dtype=torch.bfloat16
    )
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    model = get_peft_model(base, lora)

    # 加载数据集，确保路径正确
    train_ds = load_chat_jsonl(tok, f"{data_dir}/train_chat.jsonl")
    eval_ds = load_chat_jsonl(tok, f"{data_dir}/valid_chat.jsonl") if os.path.exists(f"{data_dir}/valid_chat.jsonl") else None

    # 设置训练配置
    conf = SFTConfig(
        output_dir=out_dir,
        max_length=2048,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        num_train_epochs=2,
        logging_steps=50,
        save_steps=500,
        bf16=True,
        evaluation_strategy="steps",  # 评估策略
        eval_steps=500,
        dataset_text_field="text"     # 确保数据字段为 text
    )

    # 创建训练器并开始训练
    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=conf, dataset_text_field="text"
    )
    trainer.train()
    trainer.model.save_pretrained(os.path.join(out_dir, "adapter"))

if __name__ == "__main__":
    main()

