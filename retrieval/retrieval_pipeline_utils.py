def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca:", "").strip()
    return query


