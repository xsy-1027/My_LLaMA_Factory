import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
# 设置环境变量以禁用tokenizers库的并行处理，以避免潜在的死锁问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载模型和分词器
model_path = '../models/car/qwen2_lora_sft'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def get_model_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

def test_model(test_data_path):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        conversation = item['conversations']
        question = next((c['value'] for c in conversation if c['from'] == 'human'), None)
        true_answer = next((c['value'] for c in conversation if c['from'] == 'gpt'), None)

        if question and true_answer:
            # 添加提示以规范模型回答
            prompt = f"只回答官方指导价格：{question}"
            model_answer = get_model_response(prompt)
            print(f"Question: {question}")
            print(f"Model Answer: {model_answer}")
            print(f"True Answer: {true_answer}")
            print("")

if __name__ == "__main__":
    test_data_path = '../data/car_conversations_test.json'
    test_model(test_data_path)
