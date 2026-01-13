import json 
def read_json(dir):
    with open(dir, 'r') as f:
        data = json.load(f)
    return data
def read_txt(dir): 
    with open(dir, 'r') as f:
        data = f.readlines()
    return data