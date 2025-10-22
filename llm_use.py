from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
import functools
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
    
    prompt = """
    You are an expert in correcting typographical errors in user queries. 
    Given the following user input, identify and correct any typos to improve clarity and accuracy.
    there are list of schools that the user can mistype correct them also 
    the list are 
    'ESLSCA Business School - Planeta Group',
    'École Supérieure de Tourisme (Yschools)',
    'Toulouse Business School',
    'IGEFI - FIGS Group',
    'KEDGE Business School',
    'Rennes School of Business',
    'ICN Business School',
    'EFAP - EDH Group',
    'SKEMA Business School',
    'Paris School Of Technology & Business - GALILEO Group',
    'Cours Florent - GALILEO Group',
    'EDC Business School',
    'Ecole Bleue - EDH Group',
    'WIS - FIGS Group',
    'Montpellier Business School',
    'IIM Digital School - Pôle Léonard de Vinci',
    'École Supérieure de Design (Yschools)',
    'South Champagne Business School (Yschools)',
    'Atelier De Sèvres - GALILEO Group',
    'European Business School Paris - GA Education Group',
    'Clermont School of Business',
    'Sup De Com - FIGS Group',
    'The American Business School - IGENSIA Group',
    'ESG Immobilier (GALILEO Group)',
    'ESARC Evolution - GALILEO Group',
    'Ecole 89 - GA Education Group',
    'Institut National De Gemmologie - AD Education Group',
    'Albert School',
    'CY Tech',
    'Sup De Luxe - Planete Group',
    'Audencia Business School',
    '3A - FIGS Group',
    'Narratiiv - GALILEO Group',
    '3W Academy - EDH Group',
    'IMT Atlantique',
    'EFREI',
    'HEAD School of law',
    'HETIC - GALILEO Group',
    'ESG Nantes - GALILEO Group',
    'IPSA',
    'ESTP Engineering School',
    'Brassart - EDH Group',
    'ISEP Engineering School',
    'ICD - IGENSIA Group',
    'ISTEC Business School',
    'IFA Paris - EDH Group',
    'JUNIA: HEI',
    'Sports Management School - Planeta Group',
    'EXCELIA Group',
    'ESIEE ',
    'JUNIA: ISEN',
    'Sup Career - OMNES Group',
    'ELIJE - GALILEO Group',
    'ÉSEC - EDH Group',
    'ICART - EDH Group',
    'IFAG - FIGS Group',
    'Istituto Marangoni - GALILEO Group',
    'LISAA - GALILEO Group',
    'Merkure Business School - GALILEO Group',
    'ESG Bordeaux - GALILEO Group',
    'CREAD - EDH Group',
    'emlyon Business School',
    'CLCF - GALILEO Group',
    'ESG ACT (GALILEO Group)',
    'International University of Monaco - OMNES Group',
    'CEFAM - FIGS Group',
    'IGS RH - IGENSIA Group',
    'ESMD - FIGS Group',
    'IEFT - FIGS Group',
    'ESG Rouen - GALILEO Group',
    'ISC Paris',
    'Ecoles Ferrières - GA Education Group',
    'ESILV - Pôle Léonard de Vinci',
    'EM Strasbourg Business School',
    'ISCPA - IGENSIA Group',
    'ESAIL - FIGS Group',
    'IMSI - IGENSIA Group',
    'ESG RH - GALILEO Group',
    'Bellecour - GALILEO Group',
    'Institut Culinaire de France - GALILEO Group',
    'ITM Paris - GALILEO Group',
    'ESD Ecole Supérieur du Digital - AD Education Group',
    'ESP Ecole supérieur de Publicité - AD Education Group',
    'BESIGN School ',
    'ECE Engineering School - OMNES Group',
    'ESG Tourisme - GALILEO Group',
    'MBA ESG - GALILEO Group',
    'Paris School of Business',
    'ESG Strasbourg - GALILEO Group',
    'Burgundy School of Business',
    'CESI Paris Engineering School',
    'ECAM La Salle Lyon ',
    'aivancity',
    'La Web School - GALILEO Group',
    'ESG Aix-En-Provence - GALILEO Group',
    'IHEDREA - FIGS Group',
    'EM Normandie Business School',
    'EFJ - EDH Group',
    'Brest Business School',
    'FFOLLOZZ - IGENSIA Group',
    'IDRAC - FIGS Group',
    'Digital Campus - GALILEO Group',
    'CIFACOM - GALILEO Group',
    'ESG LUXE (GALILEO Luxe)',
    'Strate School of Design - GALILEO Group',
    'Ecole de Conde - AD Education Group',
    'IMT Business School',
    'EPSI - FIGS Group',
    'ESG Finance - GALILEO Group',
    'ESAIP Engineering Sch',
    'ESG Tours - GALILEO Group',
    'IET - FIGS Group',
    'IESEG School of Management',
    'MOPA - EDH Group',
    'NEOMA Business School',
    'ESG Toulouse - GALILEO Group',
    'Atelier Chardon Savard - GALILEO Group',
    'INSEEC - OMNES Group',
    'ESCE International Business School - OMNES Group',
    'EPITA Engineering School',
    'JUNIA: ISA',
    'Sup Biotech',
    'ESGCI - GALILEO Group',
    'HEIP - OMNES Group',
    'EAC - AD Education Group',
    'RUBIKA',
    'ESSCA Business School',
    'EMLV Business School',
    'IMIS - IGENSIA Group',
    'IPI - IGENSIA Group',
    'ESG Rennes (Galileo Group)',
    'ESG Lyon - GALILEO Group',
    'IESA arts & Culture - GALILEO Group',
    'IPAG Business School',
    'ESAM - IGENSIA Group',
    'Ferrandi Paris',
    'Ecole Supérieur du Parfum - AD Education Group',
    'ESG Montpellier - GALILEO Group',
    'ESG Sport - GALILEO Group',
    'Sup de Pub - OMNES Group',
    'ILERI - FIGS Group',
    'PENNINGHEN - GALILEO Group',
    'ESDES Business School'
    User Input: {user_input}
    if the user input is empty write a query that describe the search kwargs for example if the search kwargs has program type law return "schools and programs in law" 
    {search_kwargs}
    Please provide the corrected version of the user input only without any added text.
    """
    print(search_kwargs)
    
    response = llm_4o_mini.invoke(prompt.format(user_input=user_input, search_kwargs=search_kwargs))
    print(response)
    
    return response


def relevance_check(user_input: str, doc: Document, search_kwargs: dict):
    prompt = """
    you will act as a judge to the content retrieverd from a vector database and a query 
    the task of this database is to act as a search bar so the user enters a query and the retriever return a list of schools and programs 
    your only task to check if the doc is relevant or not 
    pay attention to the names of the schools if the user entered a school name of this and the documents contains different name return False  
    and pay attention to the search kwargs of the retirever you find something not ok return False 
        'ESLSCA Business School - Planeta Group',
    'École Supérieure de Tourisme (Yschools)',
    'Toulouse Business School',
    'IGEFI - FIGS Group',
    'KEDGE Business School',
    'Rennes School of Business',
    'ICN Business School',
    'EFAP - EDH Group',
    'SKEMA Business School',
    'Paris School Of Technology & Business - GALILEO Group',
    'Cours Florent - GALILEO Group',
    'EDC Business School',
    'Ecole Bleue - EDH Group',
    'WIS - FIGS Group',
    'Montpellier Business School',
    'IIM Digital School - Pôle Léonard de Vinci',
    'École Supérieure de Design (Yschools)',
    'South Champagne Business School (Yschools)',
    'Atelier De Sèvres - GALILEO Group',
    'European Business School Paris - GA Education Group',
    'Clermont School of Business',
    'Sup De Com - FIGS Group',
    'The American Business School - IGENSIA Group',
    'ESG Immobilier (GALILEO Group)',
    'ESARC Evolution - GALILEO Group',
    'Ecole 89 - GA Education Group',
    'Institut National De Gemmologie - AD Education Group',
    'Albert School',
    'CY Tech',
    'Sup De Luxe - Planete Group',
    'Audencia Business School',
    '3A - FIGS Group',
    'Narratiiv - GALILEO Group',
    '3W Academy - EDH Group',
    'IMT Atlantique',
    'EFREI',
    'HEAD School of law',
    'HETIC - GALILEO Group',
    'ESG Nantes - GALILEO Group',
    'IPSA',
    'ESTP Engineering School',
    'Brassart - EDH Group',
    'ISEP Engineering School',
    'ICD - IGENSIA Group',
    'ISTEC Business School',
    'IFA Paris - EDH Group',
    'JUNIA: HEI',
    'Sports Management School - Planeta Group',
    'EXCELIA Group',
    'ESIEE ',
    'JUNIA: ISEN',
    'Sup Career - OMNES Group',
    'ELIJE - GALILEO Group',
    'ÉSEC - EDH Group',
    'ICART - EDH Group',
    'IFAG - FIGS Group',
    'Istituto Marangoni - GALILEO Group',
    'LISAA - GALILEO Group',
    'Merkure Business School - GALILEO Group',
    'ESG Bordeaux - GALILEO Group',
    'CREAD - EDH Group',
    'emlyon Business School',
    'CLCF - GALILEO Group',
    'ESG ACT (GALILEO Group)',
    'International University of Monaco - OMNES Group',
    'CEFAM - FIGS Group',
    'IGS RH - IGENSIA Group',
    'ESMD - FIGS Group',
    'IEFT - FIGS Group',
    'ESG Rouen - GALILEO Group',
    'ISC Paris',
    'Ecoles Ferrières - GA Education Group',
    'ESILV - Pôle Léonard de Vinci',
    'EM Strasbourg Business School',
    'ISCPA - IGENSIA Group',
    'ESAIL - FIGS Group',
    'IMSI - IGENSIA Group',
    'ESG RH - GALILEO Group',
    'Bellecour - GALILEO Group',
    'Institut Culinaire de France - GALILEO Group',
    'ITM Paris - GALILEO Group',
    'ESD Ecole Supérieur du Digital - AD Education Group',
    'ESP Ecole supérieur de Publicité - AD Education Group',
    'BESIGN School ',
    'ECE Engineering School - OMNES Group',
    'ESG Tourisme - GALILEO Group',
    'MBA ESG - GALILEO Group',
    'Paris School of Business',
    'ESG Strasbourg - GALILEO Group',
    'Burgundy School of Business',
    'CESI Paris Engineering School',
    'ECAM La Salle Lyon ',
    'aivancity',
    'La Web School - GALILEO Group',
    'ESG Aix-En-Provence - GALILEO Group',
    'IHEDREA - FIGS Group',
    'EM Normandie Business School',
    'EFJ - EDH Group',
    'Brest Business School',
    'FFOLLOZZ - IGENSIA Group',
    'IDRAC - FIGS Group',
    'Digital Campus - GALILEO Group',
    'CIFACOM - GALILEO Group',
    'ESG LUXE (GALILEO Luxe)',
    'Strate School of Design - GALILEO Group',
    'Ecole de Conde - AD Education Group',
    'IMT Business School',
    'EPSI - FIGS Group',
    'ESG Finance - GALILEO Group',
    'ESAIP Engineering Sch',
    'ESG Tours - GALILEO Group',
    'IET - FIGS Group',
    'IESEG School of Management',
    'MOPA - EDH Group',
    'NEOMA Business School',
    'ESG Toulouse - GALILEO Group',
    'Atelier Chardon Savard - GALILEO Group',
    'INSEEC - OMNES Group',
    'ESCE International Business School - OMNES Group',
    'EPITA Engineering School',
    'JUNIA: ISA',
    'Sup Biotech',
    'ESGCI - GALILEO Group',
    'HEIP - OMNES Group',
    'EAC - AD Education Group',
    'RUBIKA',
    'ESSCA Business School',
    'EMLV Business School',
    'IMIS - IGENSIA Group',
    'IPI - IGENSIA Group',
    'ESG Rennes (Galileo Group)',
    'ESG Lyon - GALILEO Group',
    'IESA arts & Culture - GALILEO Group',
    'IPAG Business School',
    'ESAM - IGENSIA Group',
    'Ferrandi Paris',
    'Ecole Supérieur du Parfum - AD Education Group',
    'ESG Montpellier - GALILEO Group',
    'ESG Sport - GALILEO Group,
    'Sup de Pub - OMNES Group',
    'ILERI - FIGS Group',
    'PENNINGHEN - GALILEO Group',
    'ESDES Business School'

    if it is relevant return True 
    if it is not rerurn False
    if the input is empty return True always
    don't show your steps answer only with True or False
    the doc {doc}
    the user_input {user_input}
    search_kwargs {search_kwargs}
    """
    
    return llm_4o_mini.invoke(f"{prompt} \n\n {prompt.format(user_input=user_input, doc=doc, search_kwargs = search_kwargs)}").content



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
    
    prompt = f"""
    You are a relevance judge for a search system that returns schools and programs.
    
    Task: For each document below, determine if it's relevant to the user query.
    
    Rules:
    - If user searches for a specific school name, only return documents from that school
    - Pay attention to the search filter requirements
    - If input is empty or blank, all documents are relevant
    - Be strict about school name matching
    
    Valid school names include:
        'ESLSCA Business School - Planeta Group',
    'École Supérieure de Tourisme (Yschools)',
    'Toulouse Business School',
    'IGEFI - FIGS Group',
    'KEDGE Business School',
    'Rennes School of Business',
    'ICN Business School',
    'EFAP - EDH Group',
    'SKEMA Business School',
    'Paris School Of Technology & Business - GALILEO Group',
    'Cours Florent - GALILEO Group',
    'EDC Business School',
    'Ecole Bleue - EDH Group',
    'WIS - FIGS Group',
    'Montpellier Business School',
    'IIM Digital School - Pôle Léonard de Vinci',
    'École Supérieure de Design (Yschools)',
    'South Champagne Business School (Yschools)',
    'Atelier De Sèvres - GALILEO Group',
    'European Business School Paris - GA Education Group',
    'Clermont School of Business',
    'Sup De Com - FIGS Group',
    'The American Business School - IGENSIA Group',
    'ESG Immobilier (GALILEO Group)',
    'ESARC Evolution - GALILEO Group',
    'Ecole 89 - GA Education Group',
    'Institut National De Gemmologie - AD Education Group',
    'Albert School',
    'CY Tech',
    'Sup De Luxe - Planete Group',
    'Audencia Business School',
    '3A - FIGS Group',
    'Narratiiv - GALILEO Group',
    '3W Academy - EDH Group',
    'IMT Atlantique',
    'EFREI',
    'HEAD School of law',
    'HETIC - GALILEO Group',
    'ESG Nantes - GALILEO Group',
    'IPSA',
    'ESTP Engineering School',
    'Brassart - EDH Group',
    'ISEP Engineering School',
    'ICD - IGENSIA Group',
    'ISTEC Business School',
    'IFA Paris - EDH Group',
    'JUNIA: HEI',
    'Sports Management School - Planeta Group',
    'EXCELIA Group',
    'ESIEE ',
    'JUNIA: ISEN',
    'Sup Career - OMNES Group',
    'ELIJE - GALILEO Group',
    'ÉSEC - EDH Group',
    'ICART - EDH Group',
    'IFAG - FIGS Group',
    'Istituto Marangoni - GALILEO Group',
    'LISAA - GALILEO Group',
    'Merkure Business School - GALILEO Group',
    'ESG Bordeaux - GALILEO Group',
    'CREAD - EDH Group',
    'emlyon Business School',
    'CLCF - GALILEO Group',
    'ESG ACT (GALILEO Group)',
    'International University of Monaco - OMNES Group',
    'CEFAM - FIGS Group',
    'IGS RH - IGENSIA Group',
    'ESMD - FIGS Group',
    'IEFT - FIGS Group',
    'ESG Rouen - GALILEO Group',
    'ISC Paris',
    'Ecoles Ferrières - GA Education Group',
    'ESILV - Pôle Léonard de Vinci',
    'EM Strasbourg Business School',
    'ISCPA - IGENSIA Group',
    'ESAIL - FIGS Group',
    'IMSI - IGENSIA Group',
    'ESG RH - GALILEO Group',
    'Bellecour - GALILEO Group',
    'Institut Culinaire de France - GALILEO Group',
    'ITM Paris - GALILEO Group',
    'ESD Ecole Supérieur du Digital - AD Education Group',
    'ESP Ecole supérieur de Publicité - AD Education Group',
    'BESIGN School ',
    'ECE Engineering School - OMNES Group',
    'ESG Tourisme - GALILEO Group',
    'MBA ESG - GALILEO Group',
    'Paris School of Business',
    'ESG Strasbourg - GALILEO Group',
    'Burgundy School of Business',
    'CESI Paris Engineering School',
    'ECAM La Salle Lyon ',
    'aivancity',
    'La Web School - GALILEO Group',
    'ESG Aix-En-Provence - GALILEO Group',
    'IHEDREA - FIGS Group',
    'EM Normandie Business School',
    'EFJ - EDH Group',
    'Brest Business School',
    'FFOLLOZZ - IGENSIA Group',
    'IDRAC - FIGS Group',
    'Digital Campus - GALILEO Group',
    'CIFACOM - GALILEO Group',
    'ESG LUXE (GALILEO Luxe)',
    'Strate School of Design - GALILEO Group',
    'Ecole de Conde - AD Education Group',
    'IMT Business School',
    'EPSI - FIGS Group',
    'ESG Finance - GALILEO Group',
    'ESAIP Engineering Sch',
    'ESG Tours - GALILEO Group',
    'IET - FIGS Group',
    'IESEG School of Management',
    'MOPA - EDH Group',
    'NEOMA Business School',
    'ESG Toulouse - GALILEO Group',
    'Atelier Chardon Savard - GALILEO Group',
    'INSEEC - OMNES Group',
    'ESCE International Business School - OMNES Group',
    'EPITA Engineering School',
    'JUNIA: ISA',
    'Sup Biotech',
    'ESGCI - GALILEO Group',
    'HEIP - OMNES Group',
    'EAC - AD Education Group',
    'RUBIKA',
    'ESSCA Business School',
    'EMLV Business School',
    'IMIS - IGENSIA Group',
    'IPI - IGENSIA Group',
    'ESG Rennes (Galileo Group)',
    'ESG Lyon - GALILEO Group',
    'IESA arts & Culture - GALILEO Group',
    'IPAG Business School',
    'ESAM - IGENSIA Group',
    'Ferrandi Paris',
    'Ecole Supérieur du Parfum - AD Education Group',
    'ESG Montpellier - GALILEO Group',
    'ESG Sport - GALILEO Group,
    'Sup de Pub - OMNES Group',
    'ILERI - FIGS Group',
    'PENNINGHEN - GALILEO Group',
    'ESDES Business School'
    
    User Query: "{user_input}"
    Search Filter: {search_kwargs}
    
    Documents to evaluate:
    {docs_text}
    
    Return ONLY a comma-separated list of document numbers that are relevant (e.g., "1,3,5" or "2,4").
    If no documents are relevant, return "NONE".
    If all documents are relevant, return "ALL".
    """
    
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