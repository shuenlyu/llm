import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TrainingArguments, Trainer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset 
import mlflow
from azureml.core import Run
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # choose the model 
    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    #load the dataset 
    # dataset = load_dataset("EleutherAI/pile", split="train[:1%]",trust_remote_code=True) #取1%的数据作为示例
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # 1% of WikiText-2
    def tokenize_function(examples):
# Tokenize inputs
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        # Set labels as shifted input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()  # Copy input_ids as labels
        return tokenized
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    #CHOOSE THE FINETUNEING METHODS
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules = ["q_proj", "v_proj"],
        lora_dropout = 0.05, 
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    #define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps (effective batch size = 4)
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets
    )
    run = Run.get_context()
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment("llm_finetuning")   
    with mlflow.start_run(nested=True):
        print(torch.cuda.memory_summary(device="cuda"))        
        trainer.train()
        
        # 5. 评估模型 (Perplexity)
        def compute_perplexity(model, tokenizer, text):

            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to("cuda")
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)  # Provide labels for consistency
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            return perplexity            
        
        sample_text = "Once upon a time, there was a large language model trained on open-source datasets."
        perplexity = compute_perplexity(model, tokenizer, sample_text)
        print(f"Perplexity: {perplexity}")
        
        mlflow.log_metric("perplexity", perplexity)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("training_epochs", training_args.num_train_epochs)
        mlflow.log_artifact("/mnt/batch/tasks/shared/LS_root/mounts/clusters/mygpuvm/code/Users/Shuen.Lyu/llm_outputs/results") 
    

if __name__ == "__main__":
    main()