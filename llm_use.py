from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import functools
from langsmith import Client
import os 

def pull_prompt_from_langsmith(prompt_name: str):
    """Pull prompt from LangSmith hub"""
    try:
        client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
        prompt = client.pull_prompt(prompt_name)
        return prompt
    except Exception as e:
        print(f"Error pulling prompt {prompt_name}: {e}")
        return None

load_dotenv()


@functools.lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

llm_4o_mini = get_llm()




def handle_typo_errors(user_input: str, search_kwargs: list ):
    # Handle empty input early
    if not user_input.strip():
        return ""
    
    prompt = pull_prompt_from_langsmith("typo-error-handle-prompt-search-bar")
    print(search_kwargs)
    
    response = llm_4o_mini.invoke(prompt.format(user_input=user_input, search_kwargs=search_kwargs)).content
    print(response)
    
    return response

def batch_relevance_filter(user_input: str, docs: list, search_kwargs: dict):
    """
    Filter a list of documents for relevance in a single LLM call.
    Returns only the relevant documents.
    """
    print(f"DEBUG: batch_relevance_filter called with user_input='{user_input}', len(docs)={len(docs)}")
    print(f"DEBUG: user_input.strip()='{user_input.strip()}', bool check={not user_input.strip()}")
    
    if not docs:
        return []
    
    # Check for empty input (including '""' case) and return all docs
    if not user_input.strip() or user_input.strip() == '""' or user_input.strip() == "''":
        print(f"Empty input detected, returning all {len(docs)} documents")
        return docs
    
    docs_text = ""
    for i, doc in enumerate(docs):
        # Truncate content to avoid token limits
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        docs_text += f"Document {i+1}:\nContent: {content_preview}\nMetadata: {doc.metadata}\n\n"
    
    prompt = pull_prompt_from_langsmith("relevance-check-search-bar").format(
            user_input=user_input,
            search_kwargs=search_kwargs,
            docs_text=docs_text
        )
    
    
    try:
        response = llm_4o_mini.invoke(prompt)
        result = response.content.strip()
        
        # Parse the response
        if result == "NONE":
            return []
        elif result == "ALL":
            return docs
        else:
            # Parse comma-separated numbers
            relevant_indices = []
            try:
                indices = [int(x.strip()) - 1 for x in result.split(',') if x.strip().isdigit()]
                relevant_indices = [i for i in indices if 0 <= i < len(docs)]
            except:
                # If parsing fails, fall back to individual checks
                print(f"Failed to parse batch response: {result}")
                return docs  # Return all docs as fallback
            
            return [docs[i] for i in relevant_indices]
            
    except Exception as e:
        print(f"Batch relevance check failed: {e}")
        # Fallback to returning all docs
        return docs