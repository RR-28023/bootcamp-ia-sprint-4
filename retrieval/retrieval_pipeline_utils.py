def modificar_query (query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query
