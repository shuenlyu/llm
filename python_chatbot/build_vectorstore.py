# -*- coding:utf-8 -*-
#----------------------------------------------
#--> File Name: build_vectorstore.py
#--> Author: lvshuen
#--> Mail: lyushuen@gmail.com 
#--> Created Time: Mon Feb 17 16:17:00 2025
#------------------------------------------------
import os 
import textwrap 

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 

def build_vectorstore(pdf_path, persist_dir):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load() 
    
    print(f"Total pages: {len(pages)}")
    
    txt_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = [] 
    for page in pages:
        content = page.page_content
        chunks = txt_splitter.split_text(content)
        for chunk in chunks:
            docs.append(chunk)
    print(f"Total documents: {len(docs)}")
    
    #使用sentence transformers做embedding
    embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedder_model)
    
    print("Building embeddings...")
    vectordb = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
if __name__ == "__main__":
    root_dir = "./docs"
    for root, dirs, files in os.walk(root_dir):
        if dirs:
            continue 
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            build_vectorstore(file_path, persist_dir="db_chroma")