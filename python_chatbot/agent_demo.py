import os 
from langchain.agents import initialize_agent, AgentType 
from langchain.agents import Tool 
from langchain.prompts import PromptTemplate 
from langchain.llms.base import LLM 
from rag_tool import create_retrival_tool
from local_llm import load_local_hf_model

def run_agent(query):
    #1. load local llm 
    # model_name = "google/flan-t5-base"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    llm = load_local_hf_model(model_name_or_path=model_name, device="cpu")
    
    #2. load retrieval tool
    retrival_tool = create_retrival_tool(persist_dir="db_chroma", k=2)
    tools = [retrival_tool]
    
    #3. customize the system promt, guide model to chain of thought
    '''
    systemp_promt = """
        you are a AI assistant who can use tool to answer questions.
        before you answer the question, you should illustrate the steps you will take to answer the question(can be briefly, which reflect chain of thoughts).
        if need outside information, you can use below tool to retrieve the information.
        1) PDFRetriever: retrive the most relevant paragraph from PDF content.
        
        Finally, provide the clear, concise and accurate answer to the question.
        ---
        if don't need any tool, you can directly answer the question.
        if need tool, please provide the Thought and Action, and summarize the answer based on Oberservation 
    """
    '''
    system_prompt = """
你是一位能使用工具的AI助理。
当你需要检索信息时，请输出如下格式：
Thought: ...
Action: PDFRetriever
Action Input: ...
Observation: ...
Thought: ...
Final Answer: ...

以下是示例对话，请严格模仿：

示例对话:
User: "我想知道苹果的营养价值"
Assistant:
Thought: 我不知道苹果的营养价值，需要查一下PDF
Action: PDFRetriever
Action Input: "苹果营养价值"
Observation: 富含维生素和膳食纤维
Thought: 已经查到相关营养信息，现在可以回答
Final Answer: 苹果富含维生素和膳食纤维

[结束示例]

现在请回答用户的问题。务必按照上述格式输出，不要省略或简写。
    
    """
    
    #4. initialize the agent
    agent = initialize_agent(
        tools = tools,
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True, 
        max_iterations = 3, 
        early_stopping_method = "generate",
        system_message = system_prompt,
        handle_parsing_errors=True
    )
    
    #5. run the agent
    answer = agent.run(query)
    return answer 

if __name__ == "__main__":
    user_questions = "please summarize the advanced use of python fucntions"
    response = run_agent(user_questions)
    print(f"\n==== Response ====\n{response}")
    