# /home/vipuser/workshop/projects/clts-sum-qwen2/scripts/testqwen2.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 本地模型路径（保持你现在的 snapshots 路径即可）===
MODEL_PATH = "/home/vipuser/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"

# === 要测试摘要的文本（随时改这段就行）===
TEXT = """请将下面一段中文材料压缩成不超过120字的要点式摘要：
Qwen2-1.5B-Instruct 是一个轻量的大语言模型，适合在消费级显卡上进行推理和微调。
它可用于文本摘要、翻译、问答、信息抽取等任务。为了得到更稳定的摘要效果，希望输出简洁、去除冗余。
"""

def main():
    # 用 bfloat16（有 GPU 时）或 float32（CPU）
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 载入本地 tokenizer 和模型（不联网）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", dtype=dtype, local_files_only=True
    )

    # 简单中文摘要指令；若模型带 chat_template，会更贴近指令风格
    system = "你是一个精炼的中文摘要助手。请抓住关键信息，语言准确、简洁。"
    user = TEXT
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system}\n\n用户：{user}\n\n请直接给出摘要："

    # 编码并推理
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 为了稳定可复现，默认不开采样；想更活泼可把 do_sample 改为 True
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,          # 稳定输出；想更发散就 True 并调 temperature/top_p
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分（不把提示词也打印出来）
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    summary = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print("\n========= 摘要结果 =========\n")
    print(summary)
    print("\n===========================\n")

if __name__ == "__main__":
    main()

