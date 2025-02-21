from langchain.agents import Tool 
from langchain.docstore.document import Document 
from langchain.vectorstores import Chroma 

def create_retrival_tool(persist_dir="db_chroma", k=2):
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=None,
    )
    
    retriver = vectordb.as_retriever(search_kwargs={"k": k})
    def retrieval_run(query):
        docs = retriver.get_relevant_documents(query)
        contents = "\n".join([doc.page_content for doc in docs])
        return f"=== Retrieved Documents ===\n{contents}"
    tool = Tool(name="PDFRetriever",
                func=retrieval_run,
                description="Retrieve documents from a vectors database.")
    return tool 