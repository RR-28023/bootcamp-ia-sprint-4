def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_2_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()

    if query.startswith("un"):
        query = query.replace("un","",1)
    elif query.startswith("una"):
        query = query.replace("una","",1)

    return query
