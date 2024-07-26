from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载模型和分词器
model_path = '../models/car/qwen2_lora_sft'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 问题
question = "一汽奥迪 奥迪A4L 2023款 40 TFSI 时尚动感型 的官方指导价是多少？"
# 编码输入
inputs = tokenizer(question, return_tensors="pt")

# 生成回答
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

# 解码输出
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 提取价格部分
import re
price_match = re.search(r'([\d.]+万)', answer)
if price_match:
    price = price_match.group(1)
else:
    price = "未找到价格"

# 打印问题和回答
print(f"问题：{question}")
print(f"回答：{price}")