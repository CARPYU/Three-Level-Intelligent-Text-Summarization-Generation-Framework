import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置本地模型路径（指向 snapshots 文件夹）
model_id = "/home/vipuser/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"  # 本地缓存路径

# 加载 Tokenizer 和 模型
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True)

# 输入示例文本
input_text = """Qwen2-1.5B-Instruct is a powerful language model capable of understanding and generating human-like text. 
It can be used for various NLP tasks such as text summarization, translation, and question answering."""

# 对输入文本进行编码
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 将输入数据移动到模型所在的设备（例如 GPU）
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# 生成文本
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # 控制生成的最大长度
        do_sample=True,      # 启用采样生成
        temperature=0.7,     # 控制生成的随机性
        top_p=0.9,           # 控制生成的多样性
        repetition_penalty=1.1,  # 避免重复生成
        eos_token_id=tokenizer.eos_token_id
    )

# 解码输出结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:", generated_text)

