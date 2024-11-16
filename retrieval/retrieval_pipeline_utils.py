def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

# añadir cualquier funcion para manejar la query y usarla en el config.py en query_prepro_fn
def mod_query(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()

    return query