from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

# 加载模型和分词器
model_path = '../models/car/qwen2_lora_sft'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# question = "奥迪 奥迪Q5L 2024款 40 TFSI 时尚动感型 的基本参数是什么？"
# question = "奥迪 奥迪Q5L 2024款 40 TFSI 豪华致雅型 的基本参数是什么？"
# question = "奥迪 奥迪Q5L 2024款 40 TFSI 豪华动感型 的基本参数是什么？"
question = "奥迪 奥迪Q5L 2025款 45 TFSI 臻选动感型 的基本参数是什么？"
#question = "奥迪 奥迪Q5L 2024款 45 TFSI 豪华动感型 的基本参数是什么？"

# 编码输入
inputs = tokenizer(question, return_tensors="pt")

start_time = time.time()

# 生成回答
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

prefilling_time = time.time() - start_time
start_time = time.time()

# 解码输出
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)

decoding_time = time.time() - start_time
print("----------------------------------------------------------------------")
print(f"Prefilling时长：{prefilling_time:.4f}秒")
print(f"Decoding时长：{decoding_time:.4f}秒")
