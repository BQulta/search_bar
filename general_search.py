from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import read_json
from filters import *
import json
from ranking import hybrid_retrieve

from llm_use import relevance_check, handle_typo_errors, batch_relevance_filter
embedding_function = HuggingFaceEmbeddings(model="intfloat/e5-large-v2")
vdb = Chroma(persist_directory="no_archive_db/", embedding_function=embedding_function)

def search(user_input: str, search_filter: str, school_ids: list, program_ids: list, more_flag: bool, is_filter_query: bool, filter_statements: list):
    
    search_kwargs = {
        "k": 15,  
        "fetch_k": 30,  
        "lambda_mult": 0.4,
    }
    
    # Apply user filters first
    if filter_statements:
        print(f"Processing filter statements: {filter_statements}")
        user_filters = filters(filter_statements)
        print(f"Generated user filters: {user_filters}")
        
        # Separate metadata filters from document filters
        metadata_filters = []
        document_filters = []
        
        for filter_condition in user_filters:
            if "where_document" in filter_condition:
                document_filters.append(filter_condition["where_document"])
            else:
                metadata_filters.append(filter_condition)
        
        print(f"Metadata filters: {metadata_filters}")
        print(f"Document filters: {document_filters}")
        
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
                search_kwargs['filter'] = {"$and": [search_kwargs['filter'], exclude_filter]}
            else:
                search_kwargs['filter'] = exclude_filter
    
    if is_filter_query == True:
        not_exclude_filter_statments = not_exclude_ids(school_ids, program_ids)
        if not_exclude_filter_statments:
            if 'filter' in search_kwargs:
                search_kwargs['filter'] = {"$and": [search_kwargs['filter'], not_exclude_filter_statments]}
            else:
                search_kwargs['filter'] = not_exclude_filter_statments
    
    # Apply school ranking filter
    if search_filter == 'schools':
        if 'filter' in search_kwargs:
            search_kwargs['filter'] = {"$and": [search_kwargs['filter']]}

    print("=== FINAL SEARCH KWARGS ===")
    print(json.dumps(search_kwargs, indent=2))
    print("===========================")
    
    retriever = vdb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    rewritten_query = handle_typo_errors(user_input, search_kwargs)
    print(f"Original query: {user_input}")
    print(f"Rewritten query: {rewritten_query}")

    school_parent_data = read_json("school_parent_json.json")
    program_parent_data = read_json("program_parent_json.json")

    # ============================================
    # 🧠 HYBRID RANKING ONLY FOR SCHOOLS
    # ============================================
    if search_filter == 'schools':
        print("Applying hybrid similarity + global rank sorting...")
        if rewritten_query == '':
            k = 1000
        else:
            k = 30 
        content = hybrid_retrieve(vdb = vdb, query= rewritten_query, k = k,filter = search_kwargs.get("filter", None))
        # print("Top 5 hybrid-ranked documents:")
        # for i, d in enumerate(content[:5]):
        #     print(f"{i+1}. rank={d.metadata.get('rank')}, sim={d.metadata.get('similarity_score'):.3f}, hybrid={d.metadata['hybrid_score']:.3f}")
    else:
        # Use normal retriever for programs or all
        content = retriever.invoke(rewritten_query)
        print(f"Raw retriever returned {len(content)} documents")
    # ============================================

    # Relevance filter
    relevant_docs = batch_relevance_filter(rewritten_query, content, search_kwargs)
    print(f"After relevance filtering: {len(relevant_docs)} documents")

    return_docs = []
    generated_school_ids = []
    generated_program_ids = []

    unique_school_ids = set() 
    unique_program_ids = set()
    
    for i, doc in enumerate(relevant_docs):
        print(f"Processing relevant doc {i}")
        school_id = doc.metadata.get('school_id')
        program_id = doc.metadata.get('program_id')

        if search_filter == 'schools':
            if school_id and school_id not in unique_school_ids:
                try:
                    school_data = school_parent_data[school_id]
                    return_docs.append(school_data)
                    generated_school_ids.append(school_id)
                    unique_school_ids.add(school_id)
                    print(f"Added school: {school_id}")
                except KeyError:
                    print(f"School {school_id} not found in parent data")

        elif search_filter == 'programs':
            if program_id and program_id not in unique_program_ids:
                try:
                    program_data = program_parent_data[program_id]
                    return_docs.append(program_data)
                    generated_program_ids.append(program_id)
                    unique_program_ids.add(program_id)
                    print(f"Added program: {program_id}")
                except KeyError:
                    print(f"Program {program_id} not found in parent data")
                
        else:  # search_filter == 'all'
            if school_id and school_id not in unique_school_ids:
                try:
                    school_data = school_parent_data[school_id]
                    return_docs.append(school_data)
                    generated_school_ids.append(school_id)
                    unique_school_ids.add(school_id)
                    print(f"Added school: {school_id}")
                except KeyError:
                    print(f"School {school_id} not found in parent data")
            
            if program_id and program_id not in unique_program_ids:
                try:
                    program_data = program_parent_data[program_id]
                    return_docs.append(program_data)
                    generated_program_ids.append(program_id)
                    unique_program_ids.add(program_id)
                    print(f"Added program: {program_id}")
                except KeyError:
                    print(f"Program {program_id} not found in parent data")

    print(f"Final results: {len(return_docs)} documents")
    print(f"School IDs: {generated_school_ids}")
    print(f"Program IDs: {generated_program_ids}")
    
    # Sort schools by rank if needed (fallback)
    if search_filter == 'schools':
        return_docs.sort(key=lambda x: (x.get('rank') is None, -(x.get('rank') or 0)))

    return return_docs, generated_school_ids, generated_program_ids, content