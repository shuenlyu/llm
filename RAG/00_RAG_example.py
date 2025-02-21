from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI 
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA 
from langchain.document_loaders import TextLoader, PyPDFLoader,UnstructuredHTMLLoader

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 

import os 

def load_documents():
    doc_dirs = "./docs"
    docs = []
    for filename in os.listdir(doc_dirs):
        filepath = os.path.join(doc_dirs, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath, encoding='utf-8')
            docs = loader.load()
            docs.extend(docs)
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(filepath, encoding='utf-8')
            docs = loader.load()
            docs.extend(docs)
        else:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load() 
            docs.extend(docs)
    return docs 

def build_embedding_model():
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return hf_embedding

def build_llm():
    model_id = "google/flan-t5-base"
    # 例子：使用 Falcon-7B-Instruct (需要GPU, 大约需要16GB显存)
    # model_id = "tiiuae/falcon-7b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 torch_dtype="auto")
    # 注意：如果是 encoder-decoder 模型（如 T5、BART），也可使用 "text2text-generation" 模式
    # 若是常见 causal LM (GPT 类结构)，用 "text-generation"。
    # Flan-T5 是 encoder-decoder，所以 pipeline 任务可以用 "text2text-generation"
    
    generate_pipeline = pipeline(task="text2text-generation", model=model,
                                 tokenizer=tokenizer, 
                                 max_length=512, 
                                 temperature=0.1,
                                 do_sample=False,)
    local_llm = HuggingFacePipeline(pipeline=generate_pipeline)
    return local_llm 
    

def build_qa_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    #构建embeddings
    hf_embedding = build_embedding_model()
    vectorstore = FAISS.from_documents(split_docs, hf_embedding)
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                           chain_type="stuff",
                                           retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                           return_source_documents = True)
    return qa_chain 

def main():
    documents = load_documents()
    qa_chain = build_qa_chain(documents)
    
    while True:
        query = input("\nPlease type your question(or type 'quit' exit): ")
        if query.lower() in ["quit", "exit"]:
            break
        
        #执行查询
        result = qa_chain({"query": query})
        answer = result['result']
        source_docs = result['source_documents']
        
        print(f"\nAnswer: {answer}")
        # 如果需要查看引用文档片段，可打印:
        for i, doc in enumerate(source_docs):
            print(f"--- Source {i+1} ---")
            print(f"文件来源: {doc.metadata.get('source')}")
            print(f"内容片段: {doc.page_content[:200]}...")
            print("-----------")
 

if __name__ == "__main__":
    main()