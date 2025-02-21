from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

summarizer = pipeline("text-generation", 
                      model=model, 
                      tokenizer=tokenizer,
                      max_length=512,
                      temperature=0.7,
                      do_sample=True)
article_text = """
北京时间 2023 年 2 月 10 日，知名科技公司发布了最新款智能手机。
该手机采用了全新的折叠屏技术，并升级了摄像头模组、芯片性能和电池续航。
业内分析师指出，这款产品有望带动新一轮的市场增长，同时也在折叠屏领域与其他厂商展开激烈竞争。
"""
# 设计 Prompt（指令），引导模型进行摘要
# Llama2 Chat 模型通常使用 [INST]...[/INST] 语法来指定系统指令、用户指令等
prompt = f"""[INST] <<SYS>>
You are a helpful assistant. Your task is to summarize the text provided by the user.
The summary should be concise, factual, and clear.
<</SYS>>

Please provide a concise summary of the following text:

{article_text}
[/INST]
"""

# 调用 pipeline 生成结果
result = summarizer(prompt, max_new_tokens=200)  # 生成的最大新 token 数可自行调节
generated_text = result[0]["generated_text"]

print("====== 原文 ======")
print(article_text.strip())
print("\n====== 摘要 ======")
print(generated_text.strip())