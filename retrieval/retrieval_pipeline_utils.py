def clean_query_txt(query: str) -> str:
    query = query.replace("El cliente busca ", "").strip()
    return query
