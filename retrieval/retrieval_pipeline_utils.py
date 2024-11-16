def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def lowercase_query(query: str) -> str: # query en minusculas
    return clean_query_txt(query).lower()

def remove_special_chars(query: str) -> str: # quita Caracteres especiales
    import re
    query = clean_query_txt(query)
    return re.sub(r"[^a-zA-Z0-9\s]", "", query)


