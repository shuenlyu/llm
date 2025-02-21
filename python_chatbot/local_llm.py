import os 
# from gpt4all import GPT4All
# from langchain.llms import GPT4All as LangChainGPT4All

import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline

def load_local_hf_model(model_name_or_path, device, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map = "auto" if device != 'cpu' else None,
        torch_dtype=torch.float16 if device == 'mps' else torch.float32
    )

    generate_pipeline = pipeline(
        "text2text-generation",
        model = model, 
        tokenizer = tokenizer,
        device = 0 if device == 'gpu' else -1,
        max_length = max_length,
        # temperature = 0.7,
        # top_p = 0.9,
        # do_sample = True,
    )
    local_llm = HuggingFacePipeline(pipeline = generate_pipeline)
    return local_llm 

def load_local_gpt4all_model(model_path, n_ctx = 512):
    local_llm = LangChainGPT4All(
        model = model_path,
        n_ctx = n_ctx,
        backend = 'gptj',
        verbose = True
    )
    return local_llm 

if __name__ == "__main__":
    # 你需要把 model_path 换成自己下载的模型
    # model_file = os.path.expanduser("~/models/ggml-gpt4all-j-v1.3-groovy.bin")
    # llm = load_local_gpt4all_model(model_file, n_ctx=512)
    # # 简单测试
    # print(llm("你好，我是GPT4All，本地模型测试"))

    # 示例：可使用一个开源的小模型做测试，如 "google/flan-t5-base" 或 "EleutherAI/gpt-neo-1.3B"
    # 如果在M1上安装了PyTorch MPS支持，可将device="mps"来尝试Metal加速
    model_name = "google/flan-t5-base"  # 或本地路径: os.path.expanduser("~/models/flan-t5/")
    llm = load_local_hf_model(model_name_or_path=model_name, device="cpu")

    # 测试一下模型交互
    prompt = "Explain the significance of chain-of-thought prompting in AI."
    print("User prompt:", prompt)
    response = llm(prompt)
    print("Model output:", response)    