#step1 define the task: seq2seq to train QA
#TODO: task1: seq2seq to train QA
import torch 
from datasets import load_dataset 
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader 
from collections import defaultdict
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 

# 3) Define a function that processes *batches* of dialogs.
def extract_qa_pairs(batch, context_size=3):
    all_inputs = []
    all_outputs = []
    # batch["dialog"] is a list of dialog lists
    for dialog in batch["dialog"]:
        for i in range(len(dialog) - 1):
            start_idx = max(0, i - context_size)
            input_context = " ".join(dialog[start_idx : i+1]).strip()
            output = dialog[i + 1].strip()
            all_inputs.append(input_context)
            all_outputs.append(output)
    return {
        "input": all_inputs,
        "output": all_outputs
    }

#tokenize the dataset 
def tokenize_function(example):
    # print(example)
    example["input"] = tokenizer(example["input"])
    example["output"] = tokenizer(example["output"])
    return example 

#construct vocab 
def yield_tokens(db_split):
    for example in db_split:
        yield example["input"]
        yield example["output"]
  
def nuericalize_tokens(example):
    example["input_ids"] = [vocab['<bos>']] + [vocab[token] for token in example["input"]] + [vocab['<eos>']]
    example["output_ids"] =[vocab['<bos>']] + [vocab[token] for token in example["output"]] + [vocab['<eos>']]
    return example

#define ptorch dataset and dataloader 
class QADataset(Dataset):
    def __init__(self, dataset, max_len=256):
        self.dataset = dataset 
        self.max_len = max_len 
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = self.dataset[idx]["input_ids"]
        output_ids = self.dataset[idx]["output_ids"]
        
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids += [vocab["<pad>"]] * (self.max_len - len(input_ids))
        
        if len(output_ids) > self.max_len:
            output_ids = output_ids[:self.max_len]
        else:
            output_ids += [vocab["<pad>"]] * (self.max_len - len(output_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long)
        }

#validate all the dataset 
#step4 define the model
def train_ngram(dataloader, n=3):
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    for batch in dataloader:
        output_ids = batch['output_ids']
        for seq in output_ids:
            tokens = [vocab.get_itos()[id] for id in seq if id != vocab['<pad>']]
            if len(tokens) < n:
                continue 
            
            #计算n-gram
            for i in range(len(tokens) - n+1):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                ngram_counts[(context, next_word)] += 1
                context_counts[context] += 1
    #calculate the probability 
    ngram_probs = {}
    for (context, word), count in ngram_counts.items():
        ngram_probs[(context, word)] = count / context_counts[context]
        
    return ngram_probs, context_counts 
#generate ngram 
def generate_ngram(ngram_probs, context_counts, input_ids, max_len=20):
    input_tokens = [vocab.get_itos()[id] for id in input_ids if id != vocab['<pad>']]
    if not input_tokens:
        context = tuple(['<bos>'])
    else:
        context = tuple(input_tokens[-(n-1):]) if len(input_tokens) >= n-1 else tuple(['<bos>'] + input_tokens)
    
    generated = list(context)
    for _ in range(max_len - len(context)):
        next_words = [(k[1], prob) for k, prob in ngram_probs.items() if k[0] == context]
        if not next_words:
            if context in context_counts:
                break 
            else:
                next_word = random.choice(vocab.get_itos()[4:]) #skip SPECIAL 
        else:
            words, probs = zip(*next_words)
            next_word = random.choices(words, weights=probs)[0]
        
        generated.append(next_word)
        context = tuple(generated[-(n-1):]) 
        if next_word == '<eos>':
            break 
    print(generated)
    return [vocab[token] for token in generated]

#测试生成
def single_test(data_loader, ngram_probs, context_counts):
    
    input_ids = data_loader[0]['input_ids'][0]
    output_ids = data_loader[0]['output_ids'][0]
    generated_ids = generate_ngram(ngram_probs, context_counts, input_ids)
    input_text = " ".join([vocab.get_itos()[id] for id in input_ids if id != vocab['<pad>']])
    output_text = " ".join([vocab.get_itos()[id] for id in output_ids if id != vocab['<pad>']])
    generated_text =  " ".join(vocab.get_itos()[id] for id in generated_ids)
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
    print(f"Generated: {generated_text}")
#step5 define the loss function and optimizer
#step6 train the model 
#step7 evaluate the model
#TODO: learn different evaluation methods
def evaluate_ngram(ngram_probs, context_counts, dataloader):
    reference = []
    hypotheses = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"]
        output_ids = batch["output_ids"]
        
        for inp, out in zip(input_ids, output_ids):
            generated_ids = generate_ngram(ngram_probs, context_counts, inp)
            reference.append([vocab.get_itos()[id] for id in out if id != vocab['<pad>']])
            hypotheses.append([vocab.get_itos()[id] for id in generated_ids if id != vocab['<pad>']])
    #calculate the BLEU score 
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(reference, hypotheses, smoothing_function=smoothing)
    return bleu_score


#step8 how good the model is 
if __name__ == "__main__":
    #step1 collect data and data preprocessing 
    #step2 collect the data
    # 1) Load the dataset
    dataset = load_dataset("daily_dialog")

    # 2) Prepare a tokenizer (if needed later)
    # TODO: tokenizer methods study
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # 3) Define a function that processes *batches* of dialogs.
    #function extract_qa_pairs is defined above
    # 4) Use batched=True, remove columns, etc.
    qa_dataset = dataset.map(
        extract_qa_pairs,
        batched=True,
        remove_columns=["act", "emotion", "dialog"]
    )

    print(qa_dataset["train"][0])
    # 5) Tokenization 
    #function tokenize_function is defined above
    tokenized_qa_dataset = qa_dataset.map(
        tokenize_function,
        batched=False,
        # remove_columns=["input", "output"]
    )
    print(tokenized_qa_dataset["train"][0])
    # 6) Build a vocabulary
    
    vocab = build_vocab_from_iterator(yield_tokens(tokenized_qa_dataset["train"]), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    print(vocab["<unk>"], vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])
    print(vocab["hello"], vocab["world"])
    # 7) Numericalize the dataset
    #function nuericalize_tokens is defined above
    numericalized_qa_dataset = tokenized_qa_dataset.map(
        nuericalize_tokens,
        batched=False,
        remove_columns=["input", "output"]
    ) 
    print(numericalized_qa_dataset["train"][0])
    # 8) Construct a PyTorch Dataset

    #define dataloader
    max_len=50
    batch_size = 16 

    train_dataset = QADataset(numericalized_qa_dataset["train"], max_len=max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = QADataset(numericalized_qa_dataset["validation"], max_len=max_len)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = QADataset(numericalized_qa_dataset["test"], max_len=max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #step2： define the model and train 
    
    #train n-gram model
    n=3      
    ngram_probs, context_counts = train_ngram(train_dataloader,n=n)
    print(f"Trained {n}-gram model with {len(ngram_probs)} n-grams")

    #testing on test dataset 
    bleu_score = evaluate_ngram(ngram_probs, context_counts, test_dataloader) 
    print(f"BLEU score: {bleu_score}")