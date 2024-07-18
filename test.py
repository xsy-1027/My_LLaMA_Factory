import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置环境变量以禁用tokenizers库的并行处理，以避免潜在的死锁问题
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = 'models/qwen_lora_dpo'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用AutoModelForCausalLM从指定路径加载模型，并将其设置为评估模式
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

# 第一轮
inputs = tokenizer.encode("你好", return_tensors="pt")    # 将文本编码为模型可以理解的输入ID
output_sequences = model.generate(inputs)   # 使用模型的generate方法生成文本序列
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)      # 将生成的文本序列解码回人类可读的文本
print(response)

# 第二轮
inputs = tokenizer.encode("给我讲一个年轻人奋斗创业最终取得成功的故事。", return_tensors="pt")
output_sequences = model.generate(inputs)
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(response)

# 第三轮
inputs = tokenizer.encode("给这个故事起一个标题。", return_tensors="pt")
output_sequences = model.generate(inputs)
response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(response)