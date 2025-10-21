from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import read_json
from llm_use import relevance_check, handle_typo_errors
from filters import *
embedding_function = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
vdb = Chroma(persist_directory="new_filter_database/", embedding_function=embedding_function)



def search(user_input: str, search_filter: str, school_ids: list, program_ids: list, more_flag: bool, is_filter_query: bool, filter_statements: list):
    

    rewritten_query = handle_typo_errors(user_input)
    
    search_kwargs = {
        "k": 10,  
        "fetch_k": 40,  
        "lambda_mult": 0.4,
    }
    
    # Apply user filters first
    if filter_statements:
        user_filters = filters(filter_statements)
        
        # Separate metadata filters from document filters
        metadata_filters = []
        document_filters = []
        
        for filter_condition in user_filters:
            if "where_document" in filter_condition:
                document_filters.append(filter_condition["where_document"])
            else:
                metadata_filters.append(filter_condition)
        
        # Apply metadata filters
        if metadata_filters:
            if len(metadata_filters) > 1:
                search_kwargs['filter'] = {"$and": metadata_filters}
            else:
                search_kwargs['filter'] = metadata_filters[0]
        
        # Apply document filters
        if document_filters:
            if len(document_filters) > 1:
                search_kwargs['where_document'] = {"$and": document_filters}
            else:
                search_kwargs['where_document'] = document_filters[0]
    
    # Apply exclusion filters
    if more_flag == True:
        exclude_filter = exclude_ids(school_ids, program_ids)
        if exclude_filter:  
            if 'filter' in search_kwargs:
                # Combine with existing metadata filters
                search_kwargs['filter'] = {"$and": [search_kwargs['filter'], exclude_filter]}
            else:
                search_kwargs['filter'] = exclude_filter
    
    if is_filter_query == True:
        not_exclude_filter_statments = not_exclude_ids(school_ids, program_ids)
        if 'filter' in search_kwargs:
            # Combine with existing metadata filters
            search_kwargs['filter'] = {"$and": [search_kwargs['filter'], not_exclude_filter_statments]}
        else:
            search_kwargs['filter'] = not_exclude_filter_statments
    
    # Apply school ranking filter
    if search_filter == 'schools':
        rank_filter = range_filter_statement('rank', 0.0, 30.0)
        if 'filter' in search_kwargs:
            # Combine with existing filters
            search_kwargs['filter'] = {"$and": [search_kwargs['filter'], rank_filter]}
        else:
            search_kwargs['filter'] = rank_filter

    print("Search kwargs:", search_kwargs)
    
    retriever = vdb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    school_parent_data = read_json("school_parent_json.json")
    program_parent_data = read_json("program_parent_json.json")
    content = retriever.invoke(rewritten_query)
    return_docs = []
    generated_school_ids = []
    generated_program_ids = []
    
    unique_school_ids = set() 
    unique_program_ids = set()

    print(f"Retrieved {len(content)} documents before relevance check")
    
    for doc in content:
        
        if relevance_check(rewritten_query, doc) == 'True':

            if search_filter == 'schools':
                school_id = doc.metadata['school_id']
                
                # Always check for uniqueness
                if school_id not in unique_school_ids:
                    school_data = school_parent_data[school_id]
                    return_docs.append(school_data)
                    generated_school_ids.append(school_id)
                    unique_school_ids.add(school_id)

            elif search_filter == 'programs':
                program_id = doc.metadata['program_id']
                
                # Always check for uniqueness
                if program_id not in unique_program_ids:
                    program_data = program_parent_data[program_id]
                    return_docs.append(program_data)
                    generated_program_ids.append(program_id)
                    unique_program_ids.add(program_id)
                    
            else:
                school_id = doc.metadata['school_id']
                program_id = doc.metadata['program_id']
                
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

                
                if is_filter_query:
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

    return return_docs, generated_school_ids, generated_program_ids, content