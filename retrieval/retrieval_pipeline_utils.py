def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = query.replace("EE.UU", "Estados Unidos")
    return query
