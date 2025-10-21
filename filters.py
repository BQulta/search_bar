def exclude_ids(school_ids, program_ids):
    filter_conditions = []
    
    if school_ids:
        filter_conditions.append({"school_id": {"$nin": school_ids}})
    
    if program_ids:
        filter_conditions.append({"program_id": {"$nin": program_ids}})
    
    if len(filter_conditions) > 1:
        return {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        return filter_conditions[0]  
    else:
        return {}  






def not_exclude_ids(school_ids, program_ids):
    filter_conditions = []
    
    if school_ids:
        filter_conditions.append({"school_id": {"$in": school_ids}})
    
    if program_ids:
        filter_conditions.append({"program_id": {"$in": program_ids}})
    
    if len(filter_conditions) > 1:
        return {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        return filter_conditions[0]  
    else:
        return {}


def numeric_filter():
    pass


def text_filter():
    pass



