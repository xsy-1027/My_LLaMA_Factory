from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

# 加载模型和分词器
model_path = '../saves/car_sft/badam'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# ------------------------------------------------------------------------------
# ques = "奥迪 奥迪Q5L 2024款 40 TFSI 豪华致雅型 的基本参数是什么？"

# ques = "奥迪系列的官方指导价 最低是多少？"

# ques = "奥迪系列的价格范围是多少？"

# ques = "奔腾B302016款长*宽*高是多少？"
# 4625*1790*1500

ques = "别克系列的君越2023款艾维亚版 多少钱啊？"
# 23.99万

# ques = "你好，今天天气怎么样？"
# ------------------------------------------------------------------------------

prompt = f"""
你是一位汽车领域AI助手，任务是回答汽车的相关问题。
----------------------------------------------
示例1：
问题：特斯拉汽车的价格是多少？
回答：特斯拉汽车的车型有很多，价格在20.69万 - 49.07万范围内。

示例2：
问题：奔腾B302016款长*宽*高是多少？
回答：4625mm*1790mm*1500mm

示例3：
问题：今天天气怎么样？
回答：今天高温38度，你需要注意防晒。
----------------------------------------------
注意，不要回答与问题无关的内容，不要重复回答!!
问题: {ques}
"""

# Prefilling 的开始时间
start_prefill = time.time()

inputs = tokenizer(prompt, return_tensors="pt")

# Prefilling 的结束时间
end_prefill = time.time()
prefill_duration = end_prefill - start_prefill

with torch.no_grad():
    start_decoding = time.time()  # Decoding开始时间

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.001,
        num_return_sequences=1,
        # repetition_penalty=2.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id

    )

    end_decoding = time.time()  # Decoding结束时间
    total_decoding_duration = end_decoding - start_decoding

    # 平均每个token的时长
    num_generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    avg_decoding_duration_per_token = total_decoding_duration / num_generated_tokens

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Prefilling 时长: {prefill_duration:.4f} 秒")
    print(f"Decoding 总时长: {total_decoding_duration:.4f} 秒")
    print(f"每个 token 的 Decoding 时长: {avg_decoding_duration_per_token:.4f} 秒")
    print(answer)

