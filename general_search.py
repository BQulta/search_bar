from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import read_json
from llm_use import relevance_check
        
def search(user_input: str, search_filter: str, school_ids: list, program_ids: list, more_flag: bool, same_query_flag: bool, filter_statements: list):
    
    embedding_function = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
    
    vdb = Chroma(persist_directory="filter_database_new/", embedding_function=embedding_function)

    search_kwargs = {
        "k": 10,  
        "fetch_k": 40,  
        "lambda_mult": 0.5,
        "$and": filter_statements
    }

    if more_flag:
        filter_statements.extend([
            {"school_id": {"$nin": school_ids}},
            {"program_id": {"$nin": program_ids}}
        ])

    print(search_kwargs)
    retriever = vdb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    school_parent_data = read_json("school_parent_json.json")
    program_parent_data = read_json("program_parent_json.json")
    content = retriever.invoke(user_input)
    return_docs = []
    generated_school_ids = []
    generated_program_ids = []
    
    unique_school_ids = set() if same_query_flag else None
    unique_program_ids = set() if same_query_flag else None

    for doc in content:
        if relevance_check(user_input, doc) == 'True':

            if search_filter == 'schools':
                
                school_id = doc.metadata['school_id']

                if same_query_flag:
                    if school_id not in unique_school_ids:
                        school_data = school_parent_data[school_id]
                        return_docs.append(school_data)
                        generated_school_ids.append(school_id)
                        unique_school_ids.add(school_id)
                else:
                    school_data = school_parent_data[school_id]
                    return_docs.append(school_data)
                    generated_school_ids.append(school_id)

            elif search_filter == 'programs':
                program_id = doc.metadata['program_id']
                
                if same_query_flag:
                    if program_id not in unique_program_ids:
                        program_data = program_parent_data[program_id]
                        return_docs.append(program_data)
                        generated_program_ids.append(program_id)
                        unique_program_ids.add(program_id)
                else:
                    program_data = program_parent_data[program_id]
                    return_docs.append(program_data)
                    generated_program_ids.append(program_id)
                    
            else:  
                school_id = doc.metadata['school_id']
                program_id = doc.metadata['program_id']
                
                if same_query_flag:
                    if school_id not in unique_school_ids:
                        school_data = school_parent_data[school_id]
                        return_docs.append(school_data)
                        generated_school_ids.append(school_id)
                        unique_school_ids.add(school_id)
                        
                    if program_id not in unique_program_ids:
                        program_data = program_parent_data[program_id]
                        return_docs.append(program_data)
                        generated_program_ids.append(program_id)
                        unique_program_ids.add(program_id)
                else:
                    school_data = school_parent_data[school_id]
                    return_docs.append(school_data)
                    program_data = program_parent_data[program_id]
                    return_docs.append(program_data)
                    generated_school_ids.append(school_id)
                    generated_program_ids.append(program_id)


    if search_filter == 'schools':
        return_docs.sort(key=lambda x: (x.get('rank') is None, -(x.get('rank') or 0)))

    return return_docs, generated_school_ids, generated_program_ids