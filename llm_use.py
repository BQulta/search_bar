from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

llm_4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)



def relevance_check(user_input: str, doc: Document):
    prompt = """
    you will act as a judge to the content retrieverd from a vector database and a query 
    the task of this database is to act as a search bar so the user enters a query and the retriever return a list of schools and programs 
    your only task to check if the doc is relevant or not 
    if it is relevant return True 
    if it is not rerurn False
    the doc {doc}
    the user_input {user_input}
    """
    
    return llm_4o_mini.invoke(f"{prompt} \n\n {prompt.format(user_input=user_input, doc=doc)}").content