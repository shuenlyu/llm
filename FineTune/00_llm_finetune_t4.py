import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TrainingArguments, Trainer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset 

def main():
    # choose the model 
    model_name = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    #load the dataset 
    dataset = load_dataset("pile", split="train[:1%]") #取1%的数据作为示例
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_lenght", max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets
    )
    
    trainer.train() 
    
    #model evaluation 
    def compute_perplexity(model, tokenizer, text):
        encodeings = tokenizer(text, return_tensors="pt")
        input_ids = encodeings.input_ids.to("cuda")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item() 
        
        return perplexity
    sample_text = "Once upon a time, there was a large language model trianed on open-source datasets."
    print(f"Perplexity: {compute_perplexity(model, tokenizer, sample_text)}")
    

if __name__ == "__main__":
    main()