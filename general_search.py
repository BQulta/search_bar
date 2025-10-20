from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import read_json
from llm_use import relevance_check

def separate_filters(filter_statements):
    """Separate document content filters from metadata filters"""
    metadata_filters = []
    document_filters = []
    
    for filter_stmt in filter_statements:
        if isinstance(filter_stmt, dict):
            # Check if this filter contains document content filters
            if contains_document_filter(filter_stmt):
                document_filters.append(filter_stmt)
            else:
                metadata_filters.append(filter_stmt)
        else:
            metadata_filters.append(filter_stmt)
    
    return metadata_filters, document_filters

def contains_document_filter(filter_dict):
    """Recursively check if filter contains where_document"""
    if isinstance(filter_dict, dict):
        for key, value in filter_dict.items():
            if key == "where_document":
                return True
            elif isinstance(value, (dict, list)):
                if contains_document_filter(value):
                    return True
    elif isinstance(filter_dict, list):
        for item in filter_dict:
            if contains_document_filter(item):
                return True
    return False

def process_document_filter(filter_dict):
    """Extract the actual filter from where_document wrapper"""
    if isinstance(filter_dict, dict):
        processed = {}
        for key, value in filter_dict.items():
            if key == "where_document":
                return value  # Return the inner filter
            elif key in ["$or", "$and"]:
                processed[key] = [process_document_filter(item) for item in value]
            else:
                processed[key] = value
        return processed
    elif isinstance(filter_dict, list):
        return [process_document_filter(item) for item in filter_dict]
    return filter_dict

def search(user_input: str, search_filter: str, school_ids: list, program_ids: list, more_flag: bool, same_query_flag: bool, filter_statements: list):
    
    embedding_function = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
    vdb = Chroma(persist_directory="filter_database_new/", embedding_function=embedding_function)

    # Create a copy to avoid modifying the original list
    all_filter_statements = filter_statements.copy()
    
    # Separate document and metadata filters
    metadata_filters, document_filters = separate_filters(all_filter_statements)
    
    # Add exclusion filters to metadata filters if more_flag is True
    if more_flag == True:
        if school_ids:  # Only add if not empty
            metadata_filters.append({"school_id": {"$nin": school_ids}})
        if program_ids:  # Only add if not empty
            metadata_filters.append({"program_id": {"$nin": program_ids}})

    # Build search_kwargs
    search_kwargs = {
        "k": 10,  
        "fetch_k": 40,  
        "lambda_mult": 0.5,
    }
    
    # Add metadata filters
    if metadata_filters:
        search_kwargs["filter"] = {"$and": metadata_filters}
    
    # Add document content filters
    if document_filters:
        # Process document filters to extract inner content
        processed_doc_filters = [process_document_filter(f) for f in document_filters]
        if len(processed_doc_filters) == 1:
            search_kwargs["where_document"] = processed_doc_filters[0]
        else:
            search_kwargs["where_document"] = {"$and": processed_doc_filters}
    
    print("Search kwargs:", search_kwargs)
    
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

    print(f"Retrieved {len(content)} documents before relevance check")
    
    for doc in content:
        # Debug: Check if excluded IDs appear in results
        doc_school_id = doc.metadata.get('school_id')
        doc_program_id = doc.metadata.get('program_id')
        
        if more_flag:
            if doc_school_id in school_ids:
                print(f"⚠️  WARNING: Excluded school_id {doc_school_id} found in results!")
            if doc_program_id in program_ids:
                print(f"⚠️  WARNING: Excluded program_id {doc_program_id} found in results!")
        
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
                    
            else:  # 'all'
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

    print(f"Generated school IDs: {generated_school_ids}")
    print(f"Generated program IDs: {generated_program_ids}")
    
    if search_filter == 'schools':
        return_docs.sort(key=lambda x: (x.get('rank') is None, -(x.get('rank') or 0)))

    return return_docs, generated_school_ids, generated_program_ids, content