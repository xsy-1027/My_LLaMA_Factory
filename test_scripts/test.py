import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置环境变量以禁用tokenizers库的并行处理，以避免潜在的死锁问题
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = '../models/car/qwen2_lora_sft'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用AutoModelForCausalLM从指定路径加载模型，并将其设置为评估模式
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

# 第一轮
inputs = tokenizer.encode("上汽奥迪 奥迪Q5 e-tron 2022款 40 e-tron 星耀型 机甲套装 的官方指导价是多少？", return_tensors="pt")    # 将文本编码为模型可以理解的输入ID
output_sequences = model.generate(inputs)
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)      # 将生成的文本序列解码回人类可读的文本
print(response)

# 第二轮
inputs = tokenizer.encode("ALPINA ALPINA B4 2016款 B4 BITURBO Coupe 的官方指导价是多少？", return_tensors="pt")
output_sequences = model.generate(inputs)
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(response)

# 第三轮
inputs = tokenizer.encode("车驰汽车 傲旋 2023款 2.0T AUXUN傲旋大白鲨 的官方指导价是多少？", return_tensors="pt")
output_sequences = model.generate(inputs)
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(response)