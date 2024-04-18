def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt_v2(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = query.replace("un", "", 1)
    query = query.replace("una", "", 1)
    
    return query