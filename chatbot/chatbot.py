from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def build_promt(user_text):
    system_text = (
        "你是一个AI聊天助手，需要尽可能准确且简洁地回答用户问题。"
        "如果对问题缺乏足够信息，请提示用户。"
    )
    prompt = f"{system_text}问题：{user_text}答案："
    return prompt

def generate_reply(user_text, max_length=512, temperature=0.7):
    prompt = build_promt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature,do_sample=True)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(reply)
    response = reply.split("答案：")[-1].strip()
    return response 

user_input = "你好，介绍一下你自己吧"
bot_response = generate_reply(user_input)
print("Chatbot: ", bot_response)