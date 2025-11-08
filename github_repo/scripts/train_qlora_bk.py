# -*- coding: utf-8 -*-
import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def load_chat_jsonl(tok, path):
    ds = load_dataset("json", data_files=path, split="train")
    def to_text(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds

def main():
    model_id = os.environ.get("MODEL_ID","Qwen/Qwen2-1.5B")
    data_dir = os.environ.get("DATA_DIR")
    out_dir  = os.environ.get("OUT_DIR","./runs/qwen2_1p5b_clts_qlora")

    assert data_dir and os.path.exists(data_dir), "Please set DATA_DIR to processed jsonl dir"

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
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
    )
    model = get_peft_model(base, lora)

    train_ds = load_chat_jsonl(tok, os.path.join(data_dir, "train_chat.jsonl"))
    eval_path = os.path.join(data_dir, "valid_chat.jsonl")
    eval_ds = load_chat_jsonl(tok, eval_path) if os.path.exists(eval_path) else None

    conf = SFTConfig(
        output_dir=out_dir,
        max_seq_length=2048, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, warmup_ratio=0.05, num_train_epochs=2,
        logging_steps=50, save_steps=500, bf16=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500
    )

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

