from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_id = '../saves/car_sft/badam'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

prompt = """
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
"""

questions = [
    "别克系列的君越2023款艾维亚版 多少钱啊？",
    "奥迪 奥迪Q5L 2024款 40 TFSI 豪华致雅型 的基本参数是什么",
    "奥迪系列的官方指导价 最低是多少？",
    "奥迪系列的价格范围是多少？",
    "奔腾B302016款长*宽*高是多少？",
    "你好，今天天气怎么样？"
]

for question in questions:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)

    print("--------------------------------")
    print("问题：", question)
    print(f"回答：{answer}")
    time.sleep(1)  # 添加短暂的延迟以避免过快的连续请求
